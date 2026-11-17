"""The ball's own say on dead ball, from the tracking alone. No tags, no training.

The ground truth starts a stoppage the moment the ball crosses a line, so the ball track is the most precise
clock we have for it: the players slow down 1-7 s later (`tracking_dead_play`) and the tags start a stoppage
when the last tagged phase ended, up to 40 s earlier (`smart_tagging.set_pieces`). Facts reported:

 1. **Out moments** (`detect_ball_out`): the first frame of each run where the tracked ball is beyond a line
    of the pitch (x outside 0..PITCH_LENGTH or y outside 0..PITCH_WIDTH), with the line it crossed and the
    first frame after it where the ball is back in play. A run shorter than MIN_OUT_FRAMES is tracking jitter.
 2. **In play** (`in_play`): per frame, the ball is inside the pitch and faster than IN_PLAY_MS. A parked ball
    (kick-off spot, goal kick, keeper's hands) and a ball being carried to a throw-in are not in play, a pass
    or a dribble is.
 3. **Lost at a goal line** (`detect_ball_lost_at_goal_line`): near the corners the track often loses the ball
    just before it crosses the goal line, so the out moment is never seen.
 4. **Restart moments** (`detect_ball_restarts`): the ball set moving after being still: a kick or a throw.
 5. **Off the pitch** (`detect_ball_off_pitch`): the ball seen well beyond a line, until it is back in play.
    Unlike the others, this alone is a stoppage, even when the tags and the players miss it.

Each fact alone is not a verdict: the ball also leaves the pitch during a throw-in that is retaken at once,
"not in play" also covers slow build-up, and the ball is missing on 41% of the graded frames, so a stoppage
the ball track does not show is still a stoppage. `resolve` combines them with the other dead-ball sources.
All thresholds are physical priors, not tuned on the public ground truth. Developed in
dev/analysis/ball_out.py, which has the checks.
"""
import numpy as np
import pandas as pd

from .. import config
from ..io import atd
from .tracking_dead_play import runs_of, speed

FPS = config.FPS

IN_PLAY_MS = 3.0  # m/s: slower than this the ball is parked, carried or rolling to a stop, not played
MIN_OUT_FRAMES = 2  # a ball beyond a line for a single frame is jitter in the track
BACK_IN_PLAY_S = 1.0  # the ball must be in play this long for the out run to be over (ignores a stray frame)

LOST_NEAR_M = 3.0  # m from a goal line: near the corners the track loses the ball 1-2 m before it crosses
LOST_MIN_S = 5.0  # the ball then stays untracked at least this long: a stoppage, not a short dropout

RESTART_MS = 6.0  # m/s: a kick or a throw sends the ball faster than this
MAX_BALL_MS = 35.0  # m/s: about the hardest shot in football; a faster jump is the track glitching, not a kick
STILL_MS = 2.0  # m/s: before a restart the ball is placed, held or carried: no tracked speed above this...
STILL_S = 1.0  # ...for at least this long (an untracked ball counts as still: it is often lost while placed)

OFF_DEEP_M = 2.0  # m beyond a line: past the track's 1-2 m error at the bench-side touchline, the ball is surely out
OFF_MIN_S = 2.0  # the ball stays out at least this long: a stoppage, not a ball skimming the line
AT_LINE_M = 1.0  # m beyond a line: a kick or throw from closer than this is a restart; from farther, a ball tossed
#                  back to the taker (ball boys, players fetching it)

EVENT_COLUMNS = ["frame", "edge", "x", "y", "back"]
LOST_COLUMNS = ["frame", "x", "y", "seconds"]
RESTART_COLUMNS = ["frame", "x", "y", "speed", "team", "dist_m"]
OFF_COLUMNS = ["start", "end", "edge", "deepest"]


def in_play(tracking: atd.Tracking) -> np.ndarray:
    """(N_FRAMES,) bool: the ball is tracked, inside the pitch and faster than IN_PLAY_MS."""
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    with np.errstate(invalid="ignore"):  # NaN compares False
        inside = (x >= 0) & (x <= config.PITCH_LENGTH) & (y >= 0) & (y <= config.PITCH_WIDTH)
        return inside & (speed(tracking.ball) > IN_PLAY_MS)


def detect_ball_out(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per time the tracked ball leaves the pitch, over the whole match.

    frame: the first frame beyond a line; edge: 'goal line' or 'sideline'; x, y: metres at that frame;
    back: the first frame after it from which the ball is in play for BACK_IN_PLAY_S (-1 if never).
    """
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    with np.errstate(invalid="ignore"):
        beyond_goal_line = (x < 0) | (x > config.PITCH_LENGTH)
        beyond_sideline = (y < 0) | (y > config.PITCH_WIDTH)
    out = (beyond_goal_line | beyond_sideline) & ~np.isnan(x)
    playing = in_play(tracking)
    hold = round(BACK_IN_PLAY_S * FPS)
    rows = []
    for s, e in runs_of(out):
        if e - s + 1 < MIN_OUT_FRAMES:
            continue
        later = np.flatnonzero(playing[e + 1:])
        back = -1
        for i in later:  # first frame from which the ball stays in play for `hold` frames
            f = e + 1 + int(i)
            if playing[f:f + hold].all() and f + hold <= len(playing):
                back = f
                break
        rows.append({"frame": int(s), "edge": "goal line" if beyond_goal_line[s] else "sideline",
                     "x": float(x[s]), "y": float(y[s]), "back": back})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).astype({"frame": int, "back": int})


def detect_ball_restarts(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per moment the ball is set moving again: faster than RESTART_MS after STILL_S without moving.

    A kick or a throw from a dead ball: the ball has been placed, held or carried (never faster than STILL_MS,
    or not tracked at all) for at least STILL_S, then it flies (but not faster than MAX_BALL_MS: that is a jump
    in the track when it finds the ball again). frame: the first fast frame; x, y: the ball
    there; speed: m/s; team, dist_m: the side of the tracked player nearest the ball there and how far he is
    (None / NaN if no player is tracked), usually the kicker or thrower. Open play also has such moments (a
    pass after the ball was briefly lost), so this is not a verdict on its own: resolve uses it only right
    after a stoppage.
    """
    v = speed(tracking.ball)
    with np.errstate(invalid="ignore"):
        fast = v > RESTART_MS
        moving = v > STILL_MS  # NaN (untracked) is False: counts as still
    still = round(STILL_S * FPS)
    rows = []
    for f in np.flatnonzero(fast[1:] & ~fast[:-1]) + 1:
        if f >= still and v[f] <= MAX_BALL_MS and not moving[f - still:f - 1].any():
            d = np.linalg.norm(tracking.players[f] - tracking.ball[f], axis=1)  # NaN: slot not tracked
            near = int(np.nanargmin(d)) if np.isfinite(d).any() else -1
            rows.append({"frame": int(f), "x": float(tracking.ball[f, 0]), "y": float(tracking.ball[f, 1]),
                         "speed": float(v[f]), "team": tracking.slot_side[near] if near >= 0 else None,
                         "dist_m": float(d[near]) if near >= 0 else np.nan})
    return pd.DataFrame(rows, columns=RESTART_COLUMNS).astype({"frame": int})


def detect_ball_lost_at_goal_line(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per time the track loses the ball next to a goal line for at least LOST_MIN_S.

    Near the corners the ball often leaves the well-tracked area just before it crosses the goal line, so
    `detect_ball_out` never sees it beyond the line. frame: the first untracked frame; x, y: the last tracked
    position, within LOST_NEAR_M of a goal line; seconds: how long the ball stays untracked. Not a verdict on
    its own (a keeper can gather a ball there): resolve uses it only where a tagged corner follows.
    """
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    rows = []
    for s, e in runs_of(np.isnan(x)):
        if s == 0 or e - s + 1 < round(LOST_MIN_S * FPS):
            continue
        lx, ly = x[s - 1], y[s - 1]
        if min(abs(lx), abs(config.PITCH_LENGTH - lx)) <= LOST_NEAR_M:
            rows.append({"frame": int(s), "x": float(lx), "y": float(ly), "seconds": (e - s + 1) / FPS})
    return pd.DataFrame(rows, columns=LOST_COLUMNS).astype({"frame": int})


def detect_ball_off_pitch(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per time the ball is clearly off the pitch: a stoppage, whatever the tags and the players say.

    Some throw-ins and goal kicks are missing from the tags, and when the players do not slow down much the
    tracking dead-play detector misses them too. But a ball seen well beyond a line is out of play. A spell
    is an out moment (`detect_ball_out`) whose ball stays out for at least OFF_MIN_S and is seen at least
    OFF_DEEP_M beyond the line. It lasts until the ball is back in play, or until an earlier restart moment
    (`detect_ball_restarts`) from inside the pitch or at most AT_LINE_M beyond it; a restart from farther out
    is the ball being tossed back to the taker. When the restart is a throw to a player's feet, the ball is
    back in play ~1-2 s after the throw, so a spell can run on that long.
    start, end: frames (end exclusive: the restart); edge: the line crossed; deepest: metres beyond it.
    """
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    with np.errstate(invalid="ignore"):
        beyond = np.max(np.c_[-x, x - config.PITCH_LENGTH, -y, y - config.PITCH_WIDTH], axis=1)  # NaN: untracked
    restarts = detect_ball_restarts(tracking)["frame"].to_numpy()
    rows = []
    for out in detect_ball_out(tracking).itertuples(index=False):
        if out.back < 0 or out.back - out.frame < round(OFF_MIN_S * FPS):
            continue
        seen = beyond[out.frame:out.back]
        if not np.isfinite(seen).any() or np.nanmax(seen) < OFF_DEEP_M:
            continue
        taken = [f for f in restarts if out.frame < f <= out.back and beyond[f] <= AT_LINE_M]
        rows.append({"start": int(out.frame), "end": int(taken[0]) if taken else int(out.back), "edge": out.edge,
                     "deepest": float(np.nanmax(seen))})
    return pd.DataFrame(rows, columns=OFF_COLUMNS).astype({"start": int, "end": int})
