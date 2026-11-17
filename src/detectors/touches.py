"""Ball touches from the tracking, and the possession they imply. Physics-based, no training.

A touch is a frame where
 1. the ball's velocity changes abruptly (kick, pass, header, deflection, trap): |v_out - v_in| > DV_MIN, and
 2. a tracked player is within DIST_MAX of the ball (+/-1 frame, a touch is often just off the closest approach),
    or within APEX_DIST_MAX of the turn's apex (`turn_apex`): the delivered ball track is heavily smoothed, so at a
    sharp turn (a pass or clearance first time, a block) it cuts the corner and never reaches the foot; the
    straight incoming and outgoing paths, extended, meet where the ball was really kicked.
Candidates closer than MIN_SEP frames are merged, keeping the strongest. The touching player is the nearest
slot, and its team gives the touch's team, except where the tracker mixes up the two keepers' tags (both
wear green): a keeper's slot in the other team's box, or an opponent's tag next to a keeper in his own box
(`fix_keeper_swaps`).

Possession from touches: the last confirmed touch team is held until the next confirmed change. A flip needs
MIN_FLIP consecutive touches by the other team, each within MAX_FLIP_GAP_S of the previous one, and is then
dated back to the first of them (`first_flip`). Touches far apart are not one hand-over: in the silence between
them the tracking has lost the ball, and an isolated stray touch must not date a later, real change back to it.
A completed pass confirms a flip after two touches already (`mark_passes`): a team that passes the ball between
two of its players controls it, which a lone touch (a tackle, a block, a deflection) does not show. Without it a
possession needs three detected touches, so the short ones (win the ball, one pass, lose it) could never exist.

Developed in dev/analysis/extract_ball_touches.py, which has the reasoning and the checks.
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd

FPS = config.FPS

K = 3  # frames either side used to estimate the velocity before / after (0.12 s)
DV_MIN = 0.2  # m/s: minimum abrupt change in ball velocity (the delivered ball track is heavily smoothed)
DIST_MAX = 1.0  # m: the touching player must be this close (foot-contact scale)
MIN_SEP = 6  # frames: two touches closer than this are merged (the stronger one is kept)
APEX_GAP = 12  # frames: the incoming / outgoing paths are read this far before / after the turn, outside the
#                smoothing (a sharp touch spreads the velocity change over ~+/-15 frames: 11551 has dv > 1 m/s
#                from 11538 to 11560)
APEX_SPAN = 10  # frames: length of each straight path
APEX_MIN_SPEED = 2.0  # m/s: both paths must move this fast, or their direction is noise
APEX_MIN_TURN = 30.0  # degrees: a smaller turn barely cuts the corner, and the paths meet ill-conditioned
APEX_MAX_CUT_M = 3.0  # m: the apex is at most this far from the smoothed ball (bounds the extrapolation)
APEX_DIST_MAX = 1.5  # m: the touching player must be this close to the apex. Over the match's sharp turns with no
#                      touch, player-to-apex distances peak at 0.75-1.0 m and the cluster ends by ~1.5-2 m; with the
#                      players taken from 5 s earlier (chance) 35 of 1035 turns pass at 1.5 m, against 359
MIN_FLIP = 3  # possession flips only after this many consecutive touches by the other team
MAX_FLIP_GAP_S = 4.0  # s: ...each this close to the previous one. A 50 m long ball at ~15 m/s flies ~3.3 s; over the
#                       match, same-team touches by different players (passes) are 1.9 s apart at the median, 3.8 s
#                       at the 75th percentile, and the tail beyond is touches the detector missed
PASS_MIN_M = 2 * DIST_MAX  # m: a pass moves the ball at least this far between the two touches. Each touching player
#                            is within DIST_MAX of the ball, so a ball that moved less could be at one player's feet
#                            the whole time, with the tracker splitting him into two slots

BOX_DEPTH = 16.5  # m: penalty area depth from the goal line (Laws of the Game)
BOX_WIDTH = 40.3  # m: penalty area width
KEEPER_SEEN = 0.5  # a keeper slot is tracked for at least this share of the half
KEEPER_MAX_AWAY = 0.03  # a keeper is in the other half at most this share of the time (a last-minute corner);
#                         a slot found there more often is shared with a field player and is not trusted
KEEPER_TWIN_M = 5.0  # m: an opponent tag this close to a trusted keeper, inside his own box, is the keeper twice
KEEPER_HISTORY_S = 10.0  # s: an attacker's tag in the box is read over this much of its past (tracked frames only)
KEEPER_SPOT_DEPTH_M = 8.0  # m: ...a median this close to the goal line, and
KEEPER_SPOT_HALF_WIDTH_M = 9.16  # m: ...this close to the goal's centre line (goal area half width) is the keeper's spot.
#                                  Attackers' tags touching in the box come from 20-50 m out; ones lingering deep stand
#                                  wide on the byline (144366, y -14.7 m) or at a corner (173320, y 15.5 m)


def nearest_player(ball: np.ndarray, players: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: distance (m) to the nearest tracked player and its slot (-1 if none)."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)  # (n, slots), NaN if either is missing
    all_nan = np.all(np.isnan(d), axis=1)
    d_filled = np.where(np.isnan(d), np.inf, d)
    slot = np.where(all_nan, -1, d_filled.argmin(axis=1))
    dist = np.where(all_nan, np.nan, d_filled.min(axis=1))
    return dist, slot


def ball_velocity_change(ball: np.ndarray, k: int = K) -> np.ndarray:
    """|v_out - v_in| (m/s) at each frame; NaN unless the ball is seen at t-k, t and t+k."""
    dv = np.full(len(ball), np.nan)
    b0, b1, b2 = ball[:-2 * k], ball[k:-k], ball[2 * k:]
    dv[k:-k] = np.linalg.norm((b2 - b1) * FPS / k - (b1 - b0) * FPS / k, axis=1)
    return dv


def turn_apex(ball: np.ndarray) -> np.ndarray:
    """(n, 2) per frame: where the straight paths into and out of a sharp turn there meet; NaN if no sharp turn.

    Incoming path: ball from t-APEX_GAP-APEX_SPAN to t-APEX_GAP; outgoing: from t+APEX_GAP to t+APEX_GAP+APEX_SPAN. Kept
    only if both move >= APEX_MIN_SPEED, turn by >= APEX_MIN_TURN, meet ahead of the incoming path and behind the
    outgoing one, and the meeting point is within APEX_MAX_CUT_M of the smoothed ball at t.
    """
    n, g, w = len(ball), APEX_GAP, APEX_SPAN
    out = np.full((n, 2), np.nan)
    t = np.arange(g + w, n - g - w)
    a1, c0 = ball[t - g], ball[t + g]
    u, v = a1 - ball[t - g - w], ball[t + g + w] - c0  # incoming / outgoing displacement over w frames
    rhs = c0 - a1
    with np.errstate(invalid="ignore", divide="ignore"):  # NaN ball and parallel paths drop out below
        det = v[:, 0] * u[:, 1] - u[:, 0] * v[:, 1]  # solve a1 + s*u = c0 + r*v
        s = (v[:, 0] * rhs[:, 1] - v[:, 1] * rhs[:, 0]) / det
        r = (u[:, 0] * rhs[:, 1] - u[:, 1] * rhs[:, 0]) / det
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        cos_turn = np.sum(u * v, axis=1) / (nu * nv)
        apex = a1 + s[:, None] * u
        ok = ((nu * FPS / w >= APEX_MIN_SPEED) & (nv * FPS / w >= APEX_MIN_SPEED)
              & (cos_turn <= np.cos(np.radians(APEX_MIN_TURN))) & (s >= 0) & (r <= 0)
              & (np.linalg.norm(apex - ball[t], axis=1) <= APEX_MAX_CUT_M))
    out[t[ok]] = apex[ok]
    return out


def detect_touches(tracking: atd.Tracking, apex: bool = True) -> pd.DataFrame:
    """One row per touch over the whole match: frame, slot, team ('H' / 'A'), dist_m, dv_ms, at_apex.

    at_apex: the touch was found at the turn's apex (`turn_apex`), not at the smoothed ball; dist_m is then the
    player's distance to the apex. apex=False switches the apex test off (ablation).
    """
    ball, players = tracking.ball, tracking.players
    dist, slot = nearest_player(ball, players)
    dv = ball_velocity_change(ball)
    dist_near = pd.Series(dist).rolling(3, center=True, min_periods=1).min().to_numpy()
    if apex:
        apex_dist, apex_slot = nearest_player(turn_apex(ball), players)
    else:
        apex_dist, apex_slot = np.full(len(ball), np.nan), np.full(len(ball), -1)

    with np.errstate(invalid="ignore"):  # NaN compares False
        at_ball = (dv > DV_MIN) & (dist_near < DIST_MAX)
        at_apex = (dv > DV_MIN) & ~at_ball & (apex_dist < APEX_DIST_MAX)

    def strongest(cand: np.ndarray) -> list[int]:
        """Non-maximum suppression: the strongest candidate wins within MIN_SEP frames."""
        keep = []
        for i in cand[np.argsort(-dv[cand])]:
            if all(abs(i - j) >= MIN_SEP for j in keep):
                keep.append(i)
        return keep

    def ball_slot(i: int) -> int:
        """The nearest slot over the +/-1 frame neighbourhood."""
        lo, hi = max(i - 1, 0), min(i + 2, len(ball))
        return int(slot[lo + int(np.argmin(np.where(np.isnan(dist[lo:hi]), np.inf, dist[lo:hi])))])

    touched = {i: ball_slot(i) for i in strongest(np.flatnonzero(at_ball))}  # frame: slot
    # apex touches are added to these, never replace them. One within APEX_GAP of a touch by the same player reads
    # the same paths as that touch, and is that touch again; by another player it is the next touch of an exchange
    ball_frames, ball_slots = np.array(sorted(touched), dtype=int), np.array([touched[j] for j in sorted(touched)])
    new = [i for i in np.flatnonzero(at_apex)
           if not np.any((np.abs(ball_frames - i) < MIN_SEP)
                         | ((np.abs(ball_frames - i) <= APEX_GAP) & (ball_slots == apex_slot[i])))]
    for i in strongest(np.array(new, dtype=int)):
        touched[int(i)] = int(apex_slot[i])
    idx = np.array(sorted(touched), dtype=int)
    touch_slot = np.array([touched[i] for i in idx], dtype=int)
    return pd.DataFrame({
        "frame": idx,
        "slot": touch_slot,
        "team": tracking.slot_side[touch_slot],
        "dist_m": np.where(at_apex[idx], apex_dist[idx], dist[idx]),
        "dv_ms": dv[idx],
        "at_apex": at_apex[idx],
    })


def keeper_slots(tracking: atd.Tracking) -> list[tuple[int, int, int, float]]:
    """Per half and team: (first frame, last frame, slot, own goal x) of the slot that is the goalkeeper.

    The keeper is the team's slot (tracked for KEEPER_SEEN of the half) whose median position is nearest an end;
    that end is its own goal. It is kept only if the slot is in the other half for at most KEEPER_MAX_AWAY of its
    tracked frames: a slot that wanders more often is sometimes a field player, and cannot be trusted.
    """
    out = []
    for lo, hi in config.HALVES.values():
        xs = tracking.players[lo:hi + 1, :, 0]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # slots never tracked in the half
            med = np.nanmedian(xs, axis=0)
        seen = (~np.isnan(xs)).mean(axis=0)
        for team in ("H", "A"):
            slots = np.flatnonzero((tracking.slot_side == team) & (seen >= KEEPER_SEEN))
            if not len(slots):
                continue
            slot = int(slots[np.argmax(np.abs(med[slots] - config.PITCH_LENGTH / 2))])
            own = 0.0 if med[slot] < config.PITCH_LENGTH / 2 else config.PITCH_LENGTH
            x = xs[:, slot]
            x = x[~np.isnan(x)]
            if (np.abs(x - own) > config.PITCH_LENGTH / 2).mean() <= KEEPER_MAX_AWAY:
                out.append((lo, hi, slot, own))
    return out


def fix_keeper_swaps(touches: pd.DataFrame, tracking: atd.Tracking) -> pd.DataFrame:
    """Fix the touches the tracker gives to the wrong team because the two keepers look alike.

    Both keepers wear green, and the tracker mixes up their tags in two ways (trusted keeper slots only,
    `keeper_slots`):
     1. when one keeper's slot has been empty for a while it hands it to the other keeper. A keeper never
        crosses the pitch into the other box in open play, so a touch there by a keeper slot is the other team's;
     2. it puts an opponent's tag on the keeper while his own tag stays nearby. A box has one keeper, so an
        opponent's touch inside the keeper's own box within KEEPER_TWIN_M of his slot is the keeper's.
    And for any slot (Athletic's keeper slot 45 is not trusted: it carries Barça's keeper for minutes at a time):
     3. an attacker's touch inside the box with no sign of an attack: the touching tag spent its last
        KEEPER_HISTORY_S at the keeper's spot (median within KEEPER_SPOT_DEPTH_M of the goal line and
        KEEPER_SPOT_HALF_WIDTH_M of its centre), where no attacker waits; an attacker's tag comes into the box.
        The touch is the defending team's (34:05, frame 51132: Barça's keeper with an Athletic tag).
    """
    from .keeper_hold import defending_teams  # keeper_hold does not import this module; kept local all the same

    out = touches.copy()
    other = {"H": "A", "A": "H"}
    in_box = lambda x, y, goal: abs(x - goal) <= BOX_DEPTH and abs(y - config.PITCH_WIDTH / 2) <= BOX_WIDTH / 2
    sides = defending_teams(tracking)
    look = int(KEEPER_HISTORY_S * FPS)
    for half, (lo, hi) in config.HALVES.items():  # 3. a keeper behind an attacker's tag
        for i in out.index[(out.frame >= lo) & (out.frame <= hi)]:
            frame, slot = out.at[i, "frame"], out.at[i, "slot"]
            x, y = tracking.players[frame, slot]
            for goal, keeper_team in zip((0.0, config.PITCH_LENGTH), sides[half]):
                if out.at[i, "team"] == keeper_team or not in_box(x, y, goal):  # NaN slot: not in the box
                    continue
                past = tracking.players[max(frame - look, lo):frame + 1, slot]
                past = past[~np.isnan(past[:, 0])]
                if (np.median(np.abs(past[:, 0] - goal)) <= KEEPER_SPOT_DEPTH_M
                        and np.median(np.abs(past[:, 1] - config.PITCH_WIDTH / 2)) <= KEEPER_SPOT_HALF_WIDTH_M):
                    out.at[i, "team"] = keeper_team
    for lo, hi, slot, own in keeper_slots(tracking):
        team = str(tracking.slot_side[slot])
        far_goal = config.PITCH_LENGTH - own
        span = (out.frame >= lo) & (out.frame <= hi)
        for i in out.index[span & (out.slot == slot)]:  # 1. the keeper's tag in the other box
            x, y = tracking.players[out.at[i, "frame"], slot]
            if in_box(x, y, far_goal):
                out.at[i, "team"] = other[team]
        for i in out.index[span & (out.team != team)]:  # 2. an opponent's tag on the keeper, in his own box
            frame = out.at[i, "frame"]
            x, y = tracking.players[frame, out.at[i, "slot"]]
            kx, ky = tracking.players[frame, slot]
            if in_box(x, y, own) and np.hypot(x - kx, y - ky) <= KEEPER_TWIN_M:  # NaN keeper: False
                out.at[i, "team"] = team
    return out


def mark_passes(touches: pd.DataFrame, tracking: atd.Tracking) -> pd.DataFrame:
    """The touch table plus `pass_in`: True where the touch receives a pass from the previous touch.

    A pass: the previous touch is by the same team but another slot, at most MAX_FLIP_GAP_S earlier, both slots
    are tracked at both touches (two people at once, not one track handed to a new slot), and the ball moved at
    least PASS_MIN_M between the touches. Run after `fix_keeper_swaps`, which can change a touch's team.
    """
    out = touches.copy()
    frames, team, slot = out.frame.to_numpy(), out.team.to_numpy(), out.slot.to_numpy()
    prev, cur = np.arange(len(out) - 1), np.arange(1, len(out))
    f0, f1 = frames[prev], frames[cur]
    s0, s1 = slot[prev], slot[cur]
    tracked = ~np.isnan(np.stack([tracking.players[f0, s0], tracking.players[f1, s0],
                                  tracking.players[f0, s1], tracking.players[f1, s1]])).any(axis=(0, 2))
    ball_m = np.linalg.norm(tracking.ball[f1] - tracking.ball[f0], axis=1)
    with np.errstate(invalid="ignore"):  # NaN ball: not a pass
        is_pass = ((team[cur] == team[prev]) & (s1 != s0) & (f1 - f0 <= MAX_FLIP_GAP_S * FPS)
                   & tracked & (ball_m >= PASS_MIN_M))
    out["pass_in"] = np.concatenate(([False], is_pass))
    return out


def first_flip(frames: np.ndarray, other: np.ndarray, min_flip: int = MIN_FLIP,
               pass_in: np.ndarray | None = None, weak: np.ndarray | None = None) -> int:
    """Index of the first touch of the first confirmed flip, or -1 if there is none.

    frames: the touches' frames, in order; other: True where the touch is by the team not in possession;
    pass_in: True where the touch receives a pass from the previous one (`mark_passes`), None for no pass rule;
    weak: True where the touch was found only at a turn's apex (`at_apex`), None if none is.
    A flip is MIN_FLIP consecutive `other` touches, each within MAX_FLIP_GAP_S of the previous one, or fewer once
    one of them (not the first) receives a pass inside the run. Weak touches do not count towards MIN_FLIP: a
    sharp turn alone is a block, a deflection or a clearance, and a player re-detected in a duel would reach
    MIN_FLIP on his own; they still take part in the run, so a pass from one confirms and dates the flip.
    """
    max_gap = MAX_FLIP_GAP_S * FPS
    cnt, strong, start = 0, 0, 0
    for i in range(len(frames)):
        if not other[i]:
            cnt = 0
            continue
        if cnt and frames[i] - frames[i - 1] > max_gap:  # too long a silence: the run starts again here
            cnt = 0
        if cnt == 0:
            start, strong = i, 0
        cnt += 1
        strong += not (weak is not None and weak[i])
        if strong >= min_flip or (cnt >= 2 and pass_in is not None and pass_in[i]):
            return start
    return -1


def touch_possession(touches: pd.DataFrame, min_flip: int = MIN_FLIP) -> np.ndarray:
    """(N_FRAMES,) H/A over the graded halves from the touches (see the module docstring); X elsewhere.

    Uses the pass rule where the table has `pass_in` (`mark_passes`); set it to False to switch the rule off.
    Apex touches (`at_apex`) are weak (see `first_flip`).
    Before the first touch of a half, its team is back-filled to the half's first frame.
    """
    frames, team = touches.frame.to_numpy(), touches.team.to_numpy()
    pass_in = touches.pass_in.to_numpy() if "pass_in" in touches else None
    weak = touches.at_apex.to_numpy() if "at_apex" in touches else None
    teams = team.copy()
    i0 = 0
    while i0 < len(team):
        cur = team[i0]
        k = first_flip(frames[i0:], team[i0:] != cur, min_flip, None if pass_in is None else pass_in[i0:],
                       None if weak is None else weak[i0:])
        end = i0 + k if k >= 0 else len(team)
        teams[i0:end] = cur  # held until the first touch of the confirming run
        i0 = end

    out = np.full(config.N_FRAMES, "X", dtype="<U1")
    for lo, hi in config.HALVES.values():
        idx = np.flatnonzero((frames >= lo) & (frames <= hi))
        if not len(idx):
            continue
        out[lo:frames[idx[0]]] = teams[idx[0]]
        for r, i in enumerate(idx):
            end = frames[idx[r + 1]] if r + 1 < len(idx) else hi + 1
            out[frames[i]:end] = teams[i]
    return out
