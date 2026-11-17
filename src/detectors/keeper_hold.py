"""The ball in a penalty box, in view of the cameras: the trace of a keeper holding it. From the tracking alone.
No tags, no training.

When the keeper has the ball in his hands (or at his feet) after a save, everyone stops running: the attackers
walk back out of the box and the defenders spread out. To the tracking dead-play detector this looks exactly
like a stoppage (slow players, ball still), but the ball is alive and the keeper's team has it.

What tells it apart from a real stoppage in the box is the ball itself. The keeper holds it inside his penalty
box, in full view of the cameras, so it stays tracked there. In a real stoppage the ball usually leaves the
pitch or is hidden among the players, and the tracking loses it.

`detect_ball_in_box` only reports, per frame, whether the ball is tracked inside a penalty box (16.5 m deep,
40.3 m wide). It is not a verdict on its own: a goal kick also has the ball still and in view in the box.
`resolve` frees a slow-players stoppage spent mostly with the ball in a box, unless the ball left the pitch
just before it or the tags place a restart inside it.

The keeper can also gather the ball out of view: he dives on it, or catches it among the players, and the track
loses it until he kicks or throws it out (37:00, frame 55514). `detect_ball_hidden_in_box` reports each time the
track loses the ball inside a box and finds it again at that same box, with the team defending it. Not a verdict
on its own either: the ball also vanishes in a crowded box while the attackers keep it (17:46, 54:49), and at a
corner or a goal. `resolve` gives the box's team the ball only when it plays it first once it is back, with no
stoppage just after.
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd
from .bench_break import runs_of
from .corner_setup import BOX_DEPTH, BOX_WIDTH

BORDER_M = 3.0  # m: the ball found again this close outside the box is still at it (the keeper's kick on its way)

HIDDEN_COLUMNS = ["start", "back", "team", "x", "y", "back_out_m"]


def detect_ball_in_box(tracking: atd.Tracking) -> np.ndarray:
    """(N_FRAMES,) bool: the ball is tracked and inside either penalty box."""
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    with np.errstate(invalid="ignore"):  # NaN (ball not tracked) compares False
        across = np.abs(y - config.PITCH_WIDTH / 2) <= BOX_WIDTH / 2
        return across & ((x <= BOX_DEPTH) | (x >= config.PITCH_LENGTH - BOX_DEPTH))


def box_distance(x: float, y: float, goal_x: float) -> float:
    """Metres from (x, y) to the penalty box at the goal line goal_x (0 inside it)."""
    dx = max(abs(x - goal_x) - BOX_DEPTH, 0.0)
    dy = max(abs(y - config.PITCH_WIDTH / 2) - BOX_WIDTH / 2, 0.0)
    return float(np.hypot(dx, dy))


def defending_teams(tracking: atd.Tracking) -> dict[int, tuple[str, str]]:
    """Map each half to (team defending the goal at x = 0, team defending the goal at x = PITCH_LENGTH).

    Each team defends the end nearer its players' median x over the half.
    """
    out = {}
    for half, (first, last) in config.HALVES.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames with no H player tracked
            home_x = np.nanmedian(tracking.players[first:last + 1, tracking.slot_side == "H", 0])
        out[half] = ("H", "A") if home_x < config.PITCH_LENGTH / 2 else ("A", "H")
    return out


def detect_ball_hidden_in_box(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per time the track loses the ball inside a penalty box and finds it again at that box.

    start: the first untracked frame, the ball's last tracked position (x, y) inside a box; back: the first
    tracked frame after, with the ball inside the same box or at most BORDER_M outside it (back_out_m);
    team: the side defending that box, i.e. the keeper's team. Only stretches starting inside a half.
    """
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    sides = defending_teams(tracking)
    rows = []
    for start, end in runs_of(np.isnan(x)):
        back = end + 1
        half = next((h for h, (first, last) in config.HALVES.items() if first <= start <= last), None)
        if half is None or back >= len(x):
            continue
        lx, ly = x[start - 1], y[start - 1]
        goal_x = 0.0 if lx < config.PITCH_LENGTH / 2 else config.PITCH_LENGTH
        if box_distance(lx, ly, goal_x) > 0:
            continue  # lost outside a box
        out_m = box_distance(x[back], y[back], goal_x)
        if out_m > BORDER_M:
            continue  # found again away from the box
        rows.append({"start": int(start), "back": int(back), "team": sides[half][0 if goal_x == 0 else 1],
                     "x": float(lx), "y": float(ly), "back_out_m": out_m})
    return pd.DataFrame(rows, columns=HIDDEN_COLUMNS).astype({"start": int, "back": int})
