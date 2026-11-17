"""Load the tracking (ATD) for the whole match, one row per tactical video frame.

Raw line i is video frame i: `frame_id:x,y,s;...(46 player slots):ball_x,ball_y,ball_s`, coordinates
normalised to 0-1, NaN where not observed. Slots are anonymous tracks with gaps and identity switches.
"""
import io
import xml.etree.ElementTree as ET
from functools import cache
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import config

N_SLOTS = 46


class Tracking(NamedTuple):
    ball: np.ndarray  # (N_FRAMES, 2) metres, x along the pitch; NaN where the ball is not seen
    players: np.ndarray  # (N_FRAMES, 46, 2) metres; NaN where the slot is not tracked
    slot_side: np.ndarray  # (46,) '<U1': 'H' / 'A', the team of each slot


@cache
def load() -> Tracking:
    """The whole match. Cached: treat the arrays as read-only."""
    paths = config.tactical_paths()
    text = paths["atd_raw"].read_text().replace(":", ",").replace(";", ",")
    raw = pd.read_csv(io.StringIO(text), header=None, na_values="NaN").to_numpy(np.float64)  # float32 shifts positions ~1e-5 m, enough to move threshold cases
    if raw.shape != (config.N_FRAMES, 1 + 3 * N_SLOTS + 3):
        raise ValueError(f"unexpected ATD shape {raw.shape}")
    scale = np.array([config.PITCH_LENGTH, config.PITCH_WIDTH])
    ball = raw[:, -3:-1] * scale
    players = raw[:, 1:-3].reshape(len(raw), N_SLOTS, 3)[:, :, :2] * scale

    root = ET.parse(paths["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]  # the first team in the metadata is the home side
    slot_side = np.array(["H" if p.get("teamId") == team_ids[0] else "A" for p in root.iter("Player")])
    if slot_side.shape != (N_SLOTS,):
        raise ValueError(f"expected {N_SLOTS} player slots in the metadata, got {slot_side.shape}")
    return Tracking(ball, players, slot_side)
