"""Breaks at the benches from the tracking alone (drinks / cooling breaks, the walk off at half time). No tags.

At a drinks break the referee stops play and both teams walk to the benches, stand there, then walk back. The
standing part is already caught by the generic dead-ball detector (players slow), but the walks are not: the
players move at ~1.3 m/s, about their open-play median, for half a minute. So:

 1. **Gathering.** Either at least BENCH_SHARE of the tracked players (>= MIN_TRACKED) stand in the bench zone
    (within BENCH_DEPTH of a touchline and BENCH_HALF_LEN of the halfway line), or fewer than FEW_TRACKED players
    are tracked at all: the others are off the pitch, at the benches or in the tunnel. Both smoothed over
    SMOOTH_S, gaps up to GATHER_GAP_S bridged, held >= MIN_GATHER_S. Open play never gets there: a throw-in
    crowds the touchline for a few seconds, a team shift stops with the median player ~15-25 m from it, and
    tracking dropouts (players hidden in a corner huddle) last a few seconds. Both touchlines are checked; the
    data does not say which one has the benches. A gathering's bench side is the touchline whose zone held it
    longest (none if it was only seen as players missing: then there are no walks to add).
 2. **Walk in.** Backwards from the gathering: the run, ending within WALK_TOL_S of the gathering start, where the
    players walk to that touchline together: median velocity towards it >= V_WALK and >= ALIGN x the median
    player speed (a slow, aligned walk; a run down the wing in open play is fast and only partly sideways).
    Gaps up to WALK_GAP_S are bridged (tracking count jumps move the median). The break starts REACTION_S before
    the walk (the whistle comes first).
 3. **Walk out.** Forwards from the gathering, the same test away from the touchline; the break ends when the
    players stop walking back (they are in position for the restart).

Developed in dev/analysis/bench_breaks.py, which has the checks. All thresholds are physical priors, not tuned on
the public ground truth (it has no break).
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd

FPS = config.FPS
HALFWAY = config.PITCH_LENGTH / 2

BENCH_DEPTH = 10.0  # m from the touchline (inside or out): the bench zone
BENCH_HALF_LEN = 30.0  # m either side of the halfway line
BENCH_SHARE = 0.5  # share of the tracked players in the bench zone
MIN_TRACKED = 4  # players tracked to judge the share (many drop out of the tracking at the benches)
FEW_TRACKED = 11  # fewer tracked players than this (half of the 22 on the pitch): the rest are off it
SMOOTH_S = 5.0  # rolling median of the share and of the tracked count
GATHER_GAP_S = 30.0  # gaps in the gathering bridged (players out of the tracked area, a short walk about)
MIN_GATHER_S = 10.0  # shortest gathering

VEL_K = 12  # frames either side for the central-difference velocity (0.96 s): a walk, not a step
WALK_SMOOTH_S = 2.0  # rolling median of the walking signals
V_WALK = 0.3  # m/s: median velocity towards (or away from) the touchline...
ALIGN = 0.5  # ...that is at least this share of the median player speed
WALK_GAP_S = 2.0  # gaps in a walk bridged
WALK_TOL_S = 5.0  # a walk must end (start) this close to the gathering's start (end)
REACTION_S = 1.0  # the players start walking ~1 s after the whistle

EVENT_COLUMNS = ["touchline_y", "start", "gather_start", "gather_end", "end"]


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of the True runs."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(s, e - 1) for s, e in zip(edges[::2], edges[1::2])]


def bridge(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than max_gap frames between two True runs."""
    out = mask.copy()
    runs = runs_of(mask)
    for (_, e), (s, _) in zip(runs[:-1], runs[1:]):
        if s - e - 1 < max_gap:
            out[e + 1:s] = True
    return out


def rolling_median(x: np.ndarray, seconds: float) -> np.ndarray:
    w = int(round(seconds * FPS)) | 1  # odd
    return pd.Series(x).rolling(w, center=True, min_periods=w // 4).median().to_numpy()


def signals(players: np.ndarray, touchline_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, for one touchline: share of the tracked players in its bench zone (NaN if too few tracked),
    median velocity towards it (m/s), median player speed (m/s). All smoothed."""
    x, y = players[..., 0], players[..., 1]
    tracked = np.isfinite(x)
    n = tracked.sum(1)
    with np.errstate(invalid="ignore"):
        in_zone = (np.abs(y - touchline_y) <= BENCH_DEPTH) & (np.abs(x - HALFWAY) <= BENCH_HALF_LEN)
    share = np.where(n >= MIN_TRACKED, in_zone.sum(1) / np.maximum(n, 1), np.nan)

    vel = np.full(players.shape, np.nan)
    vel[VEL_K:-VEL_K] = (players[2 * VEL_K:] - players[:-2 * VEL_K]) * FPS / (2 * VEL_K)
    towards = np.sign(touchline_y - config.PITCH_WIDTH / 2)  # +1: the touchline at y = PITCH_WIDTH
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where no slot is tracked
        v_towards = np.nanmedian(towards * vel[..., 1], axis=1)
        pace = np.nanmedian(np.linalg.norm(vel, axis=-1), axis=1)
    return (rolling_median(share, SMOOTH_S), rolling_median(v_towards, WALK_SMOOTH_S),
            rolling_median(pace, WALK_SMOOTH_S))


def walks(v: np.ndarray, pace: np.ndarray) -> list[tuple[int, int]]:
    """Runs where the players walk together in the direction of v (see the module docstring)."""
    with np.errstate(invalid="ignore"):
        walking = (v >= V_WALK) & (v >= ALIGN * pace)  # NaN compares False
    return runs_of(bridge(walking, int(WALK_GAP_S * FPS)))


def players_missing(players: np.ndarray) -> np.ndarray:
    """(n,) bool: fewer than FEW_TRACKED players tracked (smoothed): the others are off the pitch."""
    tracked = rolling_median(np.isfinite(players[..., 0]).sum(1).astype(float), SMOOTH_S)
    with np.errstate(invalid="ignore"):
        return tracked < FEW_TRACKED  # NaN compares False


def detect_bench_breaks(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per break: touchline_y (m, the bench side; NaN if the players were only seen missing), start
    (first dead frame: walk-in onset minus the reaction), gather_start / gather_end (inclusive, at the benches
    or off the pitch), end (exclusive: the walk back is over). Only the graded halves are searched; a break
    never leaves its half."""
    touchlines = (0.0, config.PITCH_WIDTH)
    at_bench, walks_in, walks_out = {}, {}, {}
    for t in touchlines:
        share, v_towards, pace = signals(tracking.players, t)
        with np.errstate(invalid="ignore"):
            at_bench[t] = share >= BENCH_SHARE  # NaN compares False
        walks_in[t], walks_out[t] = walks(v_towards, pace), walks(-v_towards, pace)
    away = players_missing(tracking.players) | np.logical_or.reduce(list(at_bench.values()))

    rows = []
    tol = int(WALK_TOL_S * FPS)
    for first, last in config.HALVES.values():
        in_half = np.zeros(config.N_FRAMES, dtype=bool)
        in_half[first:last + 1] = True
        for gs, ge in runs_of(bridge(away & in_half, int(GATHER_GAP_S * FPS))):
            if ge - gs + 1 < MIN_GATHER_S * FPS:
                continue
            held = {t: at_bench[t][gs:ge + 1].sum() for t in touchlines}
            bench = max(touchlines, key=held.get) if max(held.values()) else None
            walk_in = [s for s, e in walks_in[bench] if s < gs and e >= gs - tol] if bench is not None else []
            walk_out = [e for s, e in walks_out[bench] if e > ge and s <= ge + tol] if bench is not None else []
            start = walk_in[0] - int(REACTION_S * FPS) if walk_in else gs
            end = walk_out[-1] + 1 if walk_out else ge + 1
            rows.append({"touchline_y": np.nan if bench is None else bench, "start": max(start, first),
                         "gather_start": gs, "gather_end": ge, "end": min(end, last + 1)})
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events.astype({c: int for c in EVENT_COLUMNS[1:]}).sort_values("start", ignore_index=True)


def bench_mask(events: pd.DataFrame) -> np.ndarray:
    """(N_FRAMES,) bool: True over each break, walk in to walk out (the ball is dead)."""
    mask = np.zeros(config.N_FRAMES, dtype=bool)
    for start, end in events[["start", "end"]].itertuples(index=False):
        mask[start:end] = True
    return mask
