"""The main solution: turns the data into one possession label per video frame.

    load data
    run the detectors (`src/detectors/`), then resolve their answers (`src/resolve.py`)
    carry forward (`src/fill.py`), then force X outside the graded halves
"""
import numpy as np
import pandas as pd

from . import config, fill, resolve
from .detectors import (ball_out, bench_break, corner_setup, goal_celebration, keeper_hold, kick_off, touches,
                        tracking_dead_play)
from .io import atd, smart_tagging


def graded_mask() -> np.ndarray:
    """True on the frames the scorer grades."""
    mask = np.zeros(config.N_FRAMES, dtype=bool)
    for first, last in config.HALVES.values():
        mask[first:last + 1] = True
    return mask


def collect_evidence() -> dict[str, np.ndarray | pd.DataFrame]:
    """Load the data and run every detector over the whole match; keys are `resolve`'s arguments."""
    tracking = atd.load()
    tagging = smart_tagging.load()
    touch_events = touches.mark_passes(touches.fix_keeper_swaps(touches.detect_touches(tracking), tracking), tracking)
    return {
        "tags": tagging,
        "detected_dead_play": tracking_dead_play.detect_dead_play(tracking),
        "touches": touch_events,
        "touch_possession": touches.touch_possession(touch_events),
        "goal_celebrations": goal_celebration.detect_goal_celebrations(tracking),
        "bench_breaks": bench_break.detect_bench_breaks(tracking),
        "kick_offs": kick_off.detect_kick_offs(tracking),
        "ball_out": ball_out.detect_ball_out(tracking),
        "ball_in_play": ball_out.in_play(tracking),
        "ball_lost": ball_out.detect_ball_lost_at_goal_line(tracking),
        "ball_restarts": ball_out.detect_ball_restarts(tracking),
        "corner_setups": corner_setup.detect_corner_setups(tracking),
        "ball_in_box": keeper_hold.detect_ball_in_box(tracking),
        "ball_hidden_in_box": keeper_hold.detect_ball_hidden_in_box(tracking),
        "ball_off_pitch": ball_out.detect_ball_off_pitch(tracking),
    }


def predict() -> np.ndarray:
    """Labels per video frame; frames outside the graded halves end up 'X'."""
    evidence = collect_evidence()
    labels = resolve.resolve(**evidence)
    labels = fill.carry_forward(labels)
    labels[~graded_mask()] = "X"
    return labels


def run() -> pd.DataFrame:
    """Return a DataFrame with columns [frame, label], one row per video frame."""
    return pd.DataFrame({"frame": np.arange(config.N_FRAMES), "label": predict()})
