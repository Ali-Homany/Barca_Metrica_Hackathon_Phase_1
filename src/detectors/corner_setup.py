"""Set-piece setups in a penalty box, from the tracking alone: many players gathered in one box, walking.
No tags, no training.

Before a corner the attackers walk into the box and the defenders mark them, so one penalty box fills up with
players while nobody runs. The tracking dead-play detector misses most of that wait: the players walk at
1.0-1.5 m/s, about the pace of slow build-up play, and only look stopped in the last seconds before the kick.
The ball cannot help: it sits by the corner flag, where it is almost never tracked.

A setup is a stretch of at least MIN_SETUP_S where at least MIN_IN_BOX tracked players are inside one penalty
box (16.5 m deep, 40.3 m wide) and the median speed of the tracked players is below SLOW_MS. Short dropouts
(up to MAX_GAP_S) are bridged.

A setup alone is not a verdict: a full box with slow players also happens in live play, during a slow attack
that ends in a foul, or right after a corner kick while the ball is cleared. `resolve` only uses a setup to
confirm a tagged corner, and only between the ball crossing the goal line and that corner kick.
All thresholds are physical priors, not tuned on the public ground truth.
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd
from .tracking_dead_play import runs_of, speed

FPS = config.FPS

BOX_DEPTH = 16.5  # m: penalty area depth from the goal line (Laws of the Game)
BOX_WIDTH = 40.3  # m: penalty area width
MIN_IN_BOX = 10  # tracked players in one box: a corner brings ~10-16; open play rarely more than a few for long
SLOW_MS = 1.5  # m/s: median player speed below this is walking (open play runs at ~2-3 m/s)
SPEED_K = 6  # frames either side for each player's speed (0.48 s): a walk, not a step
MIN_SETUP_S = 5.0  # shortest setup
MAX_GAP_S = 1.0  # gaps shorter than this inside a setup are bridged

EVENT_COLUMNS = ["start", "end", "side", "in_box"]


def players_in_box(players: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N_FRAMES,) tracked players inside the left and the right penalty box."""
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):  # NaN compares False
        across = np.abs(y - config.PITCH_WIDTH / 2) < BOX_WIDTH / 2
        left = (across & (x >= 0) & (x < BOX_DEPTH)).sum(axis=1)
        right = (across & (x <= config.PITCH_LENGTH) & (x > config.PITCH_LENGTH - BOX_DEPTH)).sum(axis=1)
    return left, right


def detect_corner_setups(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per setup: start, end (frames, inclusive), side ('left' / 'right'), in_box (median count)."""
    left, right = players_in_box(tracking.players)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked give NaN, which is not slow
        walking = np.nanmedian(speed(tracking.players, SPEED_K), axis=1) < SLOW_MS
    gap = round(MAX_GAP_S * FPS)
    rows = []
    for side, count in (("left", left), ("right", right)):
        setup = (count >= MIN_IN_BOX) & walking
        runs = runs_of(setup)
        for (_, e0), (s1, _) in zip(runs, runs[1:]):  # bridge short gaps between runs
            if s1 - e0 - 1 <= gap:
                setup[e0 + 1:s1] = True
        for s, e in runs_of(setup):
            if e - s + 1 >= round(MIN_SETUP_S * FPS):
                rows.append({"start": int(s), "end": int(e), "side": side, "in_box": float(np.median(count[s:e + 1]))})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("start", ignore_index=True)
