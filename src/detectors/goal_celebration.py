"""Goals from the tracking alone: the celebration run, and the walk back to kick-off positions. No tags, no training.

A goal is the only thing that sends three or more players of one team sprinting to the same spot and is
followed by both teams lining up for a kick-off. So:

 1. **Convergence (the celebration run).** At a frame, the running players of one team (speed >= V_RUN) are
    projected forward along their velocity by 1, 2 and 3 s. If >= MIN_RUNNERS of them would meet within
    R_MEET of each other, the team is running into one point. Projecting the run, rather than waiting for the
    huddle itself, matters: celebrations happen at the corner flag or behind the goal, where players leave the
    tracked area or merge into one blob, so the huddle is often never seen in the tracking.
 2. **Goal moment (backtrack).** From the convergence, each runner is followed back in time for as long as it
    keeps running (>= V_ONSET) towards the meeting point (within ONSET_COS of the direction to it). The median
    of those run onsets is when the celebration started; the goal is REACTION_S before it (the players'
    reaction time after the ball goes in).
 3. **Celebration end (kick-off layout).** The first frame where both teams stand in kick-off layout: each team
    on its own side of the halfway line (>= SIDE_FRAC of its tracked players, HALFWAY_TOL slack), on the sides
    the half is played in, and spread out (median distance to the nearest teammate >= NN_MIN, which rejects
    drinks breaks and bench gatherings where each team is also on its own side). Held for >= MIN_LAYOUT_S.
 4. **Pairing.** Every kick-off layout after the half's opening one closes a celebration if a convergence
    happened since the previous layout and at most MAX_CELEB_S before it; the last such convergence is the
    celebration run. A layout with no convergence before it (a drinks break that passed the spacing test) and a
    convergence with no layout after it (pressing, a counter-attack) are both ignored.

Each half's orientation (which side H defends) is the majority of its kick-off layouts. The opening layout of a
half is searched from LEAD_S before the graded window, so the pre-kick-off team huddle can never pair with it.
All thresholds are physical priors, not tuned on the public ground truth. Developed in
dev/analysis/goal_celebrations.py, which has the reasoning and the checks.
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd

FPS = config.FPS
HALFWAY = config.PITCH_LENGTH / 2

VEL_K = 6  # frames either side for the central-difference velocity (0.48 s): smooths tracking jitter
V_RUN = 4.0  # m/s: a player is running (a celebration sprint; jogging back is ~2 m/s)
TAUS_S = (1.0, 2.0, 3.0)  # look-ahead times at which the runners' paths are tested for a common point
R_MEET = 4.0  # m: projected positions this close count as the same point
MIN_RUNNERS = 3  # players of one team running into the same point

V_ONSET = 2.0  # m/s: backtracking stops once a runner slows below this...
ONSET_COS = 0.5  # ...or stops heading to the meeting point (more than 60 degrees off)
MAX_BACK_S = 10.0  # a celebration run is never traced back further than this
REACTION_S = 1.0  # players start the celebration run ~1 s after the ball goes in; the goal is dated this much earlier

MIN_TRACKED = 7  # players per team needed to judge a layout
SIDE_FRAC = 0.8  # share of a team's tracked players that must be on its own side (one stray slot is common)
HALFWAY_TOL = 1.5  # m: the kick-off takers stand on the halfway line, inside the centre circle
NN_MIN = 4.0  # m: median distance to the nearest teammate; bench gatherings and huddles are ~1-2 m
MIN_LAYOUT_S = 2.2  # the layout must hold this long: at a real kick-off both teams wait for the whistle, while
#                     open play passes through a "layout" in 1-2 s (Metrica sample games 1-3: real kick-off waits
#                     >= 2.4 s, the passing ones behind 5 of 9 fake goals 1.2-2.0 s). BAR-ATH: the same 3 goals
#                     and 5 kick-offs as with 1 s
MAX_CELEB_S = 240.0  # from the goal to the kick-off layout; a VAR check makes it ~2.5 min
LEAD_S = 60.0  # context before each graded half, to find its opening kick-off layout

CHUNK = 20_000  # frames per block in the pairwise computations (memory)
EVENT_COLUMNS = ["team", "goal_frame", "run_frame", "meet_x", "meet_y", "n_runners", "end_frame"]


def velocity(xy: np.ndarray, k: int = VEL_K) -> np.ndarray:
    """Central-difference velocity (m/s) per frame, same shape as xy; NaN where either end is missing."""
    v = np.full(xy.shape, np.nan)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    return v


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of the True runs."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(s, e - 1) for s, e in zip(edges[::2], edges[1::2])]


def pairwise(xy: np.ndarray) -> np.ndarray:
    """(n, k, 2) points to (n, k, k) distances, NaN where either point is missing."""
    return np.linalg.norm(xy[:, :, None] - xy[:, None], axis=-1)


# --- 1. convergence -------------------------------------------------------------------------------------------

def meeting_runners(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One team. Per frame: how many runners meet at one point (best look-ahead), and that point.

    pos, vel: (n, k, 2). Returns (count (n,), point (n, 2), NaN where count < MIN_RUNNERS).
    """
    n = len(pos)
    count = np.zeros(n, dtype=int)
    point = np.full((n, 2), np.nan)
    with np.errstate(invalid="ignore"):
        running = np.linalg.norm(vel, axis=-1) >= V_RUN  # NaN compares False
    idx = np.flatnonzero(running.sum(1) >= MIN_RUNNERS)  # only these frames can pass (a few % of the match)
    for lo in range(0, len(idx), CHUNK):
        f = idx[lo:lo + CHUNK]
        for tau in TAUS_S:
            ahead = np.where(running[f, :, None], pos[f] + vel[f] * tau, np.nan)
            close = pairwise(ahead) < R_MEET  # (m, k, k), NaN compares False
            per_player = close.sum(-1)
            best = per_player.argmax(-1)
            c = per_player.max(-1)
            better = c > count[f]
            members = close[np.arange(len(best)), best]  # (m, k): the runners around the best one
            centre = np.where(members[..., None], ahead, 0.0).sum(1) / np.maximum(c, 1)[:, None]
            count[f] = np.where(better, c, count[f])
            point[f] = np.where(better[:, None], centre, point[f])
    point[count < MIN_RUNNERS] = np.nan
    return count, point


def convergences(count: np.ndarray, point: np.ndarray, team: str, first: int, last: int) -> list[dict]:
    """The runs of >= MIN_RUNNERS meeting runners inside [first, last], one dict per run."""
    out = []
    for s, e in runs_of(count[first:last + 1] >= MIN_RUNNERS):
        s, e = s + first, e + first
        peak = s + int(np.argmax(count[s:e + 1]))
        out.append({"team": team, "run_frame": s, "peak": peak, "n_runners": int(count[peak]),
                    "meet_x": point[peak, 0], "meet_y": point[peak, 1]})
    return out


# --- 2. backtrack ---------------------------------------------------------------------------------------------

def run_onset(pos: np.ndarray, vel: np.ndarray, peak: int, target: np.ndarray) -> int:
    """Median frame at which the runners at `peak` started running towards `target`. pos, vel: (n, k, 2)."""
    to_target = target - pos[peak]
    with np.errstate(invalid="ignore"):
        heading = np.einsum("kc,kc->k", vel[peak], to_target) / (
            np.linalg.norm(vel[peak], axis=-1) * np.linalg.norm(to_target, axis=-1))
        runners = np.flatnonzero((np.linalg.norm(vel[peak], axis=-1) >= V_RUN) & (heading >= ONSET_COS))
    stop = max(peak - int(MAX_BACK_S * FPS), 0)
    onsets = []
    for j in runners:
        f = peak
        while f - 1 >= stop:
            v, x = vel[f - 1, j], pos[f - 1, j]
            speed = np.linalg.norm(v)
            d = target - x
            if not (speed >= V_ONSET and v @ d >= ONSET_COS * speed * np.linalg.norm(d)):  # NaN stops the backtrack
                break
            f -= 1
        onsets.append(f)
    return int(np.median(onsets)) if onsets else peak


# --- 3. kick-off layout ---------------------------------------------------------------------------------------

def team_sides(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One team, per frame: tracked count, share of it left of halfway, share right of halfway (with slack)."""
    tracked = np.isfinite(pos[..., 0])
    n = tracked.sum(1)
    x = pos[..., 0]
    with np.errstate(invalid="ignore"):
        left = ((x < HALFWAY + HALFWAY_TOL) & tracked).sum(1) / np.maximum(n, 1)
        right = ((x > HALFWAY - HALFWAY_TOL) & tracked).sum(1) / np.maximum(n, 1)
    return n, left, right


def spacing(pos: np.ndarray) -> np.ndarray:
    """One team, per frame: median distance (m) from each tracked player to its nearest teammate."""
    k = pos.shape[1]
    nn = np.full(len(pos), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with < 2 tracked players
        for lo in range(0, len(pos), CHUNK):
            d = pairwise(pos[lo:lo + CHUNK])
            d[:, np.arange(k), np.arange(k)] = np.nan
            nn[lo:lo + CHUNK] = np.nanmedian(np.nanmin(d, axis=2), axis=1)
    return nn


def kickoff_layouts(home: np.ndarray, away: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) runs of the kick-off layout; home / away: (n, k, 2) over one half (+ lead)."""
    n_h, left_h, right_h = team_sides(home)
    n_a, left_a, right_a = team_sides(away)
    enough = (n_h >= MIN_TRACKED) & (n_a >= MIN_TRACKED)
    h_left = enough & (left_h >= SIDE_FRAC) & (right_a >= SIDE_FRAC)
    h_right = enough & (right_h >= SIDE_FRAC) & (left_a >= SIDE_FRAC)
    idx = np.flatnonzero(h_left | h_right)  # spacing only where the sides already fit (rare in open play)
    spread = np.zeros(len(home), dtype=bool)
    with np.errstate(invalid="ignore"):
        spread[idx] = (spacing(home[idx]) >= NN_MIN) & (spacing(away[idx]) >= NN_MIN)
    h_left &= spread
    h_right &= spread
    min_len = int(MIN_LAYOUT_S * FPS)
    runs_left = [r for r in runs_of(h_left) if r[1] - r[0] + 1 >= min_len]
    runs_right = [r for r in runs_of(h_right) if r[1] - r[0] + 1 >= min_len]
    length = lambda runs: sum(e - s + 1 for s, e in runs)
    return runs_left if length(runs_left) >= length(runs_right) else runs_right  # the half's orientation


# --- 4. pairing -----------------------------------------------------------------------------------------------

def detect_goal_celebrations(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per detected goal: team (the side that scored, 'H' / 'A'), goal_frame (backtracked run onset minus
    the reaction time),
    run_frame (first frame of the convergence), meet_x, meet_y (metres), n_runners, end_frame (first frame of the
    kick-off layout that ends the celebration)."""
    players, slot_side = tracking.players, tracking.slot_side
    rows = []
    for first, last in config.HALVES.values():
        lo, hi = max(first - int(LEAD_S * FPS), 0), last + 1
        pos = {t: players[lo:hi, slot_side == t] for t in ("H", "A")}
        vel = {t: velocity(p) for t, p in pos.items()}
        layouts = kickoff_layouts(pos["H"], pos["A"])
        if len(layouts) < 2:  # only the opening kick-off (or nothing): no goal in this half
            continue
        candidates = []
        for t in ("H", "A"):
            count, point = meeting_runners(pos[t], vel[t])
            candidates += convergences(count, point, t, 0, hi - lo - 1)
        candidates.sort(key=lambda c: c["run_frame"])

        for (_, prev_end), (start, _) in zip(layouts[:-1], layouts[1:]):
            window_lo = max(prev_end + 1, start - int(MAX_CELEB_S * FPS))
            before = [c for c in candidates if window_lo <= c["run_frame"] < start]
            if not before:
                continue
            c = before[-1]  # the last run into one point before the line-up is the celebration
            target = np.array([c["meet_x"], c["meet_y"]])
            goal = run_onset(pos[c["team"]], vel[c["team"]], c["peak"], target) - int(REACTION_S * FPS)
            rows.append({"team": c["team"], "goal_frame": goal + lo, "run_frame": c["run_frame"] + lo,
                         "meet_x": c["meet_x"], "meet_y": c["meet_y"], "n_runners": c["n_runners"],
                         "end_frame": start + lo})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).astype(
        {"goal_frame": int, "run_frame": int, "n_runners": int, "end_frame": int})


def celebration_mask(events: pd.DataFrame) -> np.ndarray:
    """(N_FRAMES,) bool: True from each goal to the kick-off layout that follows it (the ball is dead)."""
    mask = np.zeros(config.N_FRAMES, dtype=bool)
    for goal, end in events[["goal_frame", "end_frame"]].itertuples(index=False):
        mask[goal:end] = True
    return mask
