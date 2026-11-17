"""Kick-offs from the tracking alone: both teams lined up, and the taker on the centre spot. No tags, no training.

 1. **Layout.** `goal_celebration.kickoff_layouts`: each team on its own half and spread out, held >= MIN_LAYOUT_S (that
    module has the full test). The runs are broken by a stray slot or a player stepping over the line, so runs
    less than MERGE_GAP_S apart are one kick-off. The layout breaks ~1-2 s after the ball is played, when
    the players run forward.
 2. **Taker.** At a kick-off only the kicking team may be inside the centre circle (Laws of the Game, 8.1), and
    its taker stands next to the ball on the spot. So, over TAKER_WINDOW_S before the layout breaks, the
    taker is the team whose player is nearest the centre spot in most frames. That is 100% of the frames at
    all 5 kick-offs of the match, including the 2nd-half one, where the ball is rolled in and the H taker waits
    ~2-3 m away. The taker is None if neither team has the nearest player in MIN_TAKER_SHARE of the frames.

A drinks break or bench gathering fails the spacing test, so it never yields a kick-off here (the one tagged
'Kick Off (After Goal)' at 62097 does not). Developed from the kick-off tag audit (2026-11-13): every tagged
real kick-off restart lies inside its layout, 27-45 frames before it breaks.
"""
import numpy as np
import pandas as pd

from .. import config
from ..io import atd
from .goal_celebration import LEAD_S, kickoff_layouts  # the layout test is shared with the goal detector

FPS = config.FPS
CENTRE = np.array([config.PITCH_LENGTH / 2, config.PITCH_WIDTH / 2])

MERGE_GAP_S = 5.0  # layout runs this close are the same kick-off (gaps seen: 0.1-3.3 s)
TAKER_WINDOW_S = (6.0, 1.5)  # judge the taker from 6 s to 1.5 s before the layout breaks (the ball is played ~1-2 s before)
MIN_TAKER_SHARE = 2 / 3  # share of those frames the taker's team must have the player nearest the spot

COLUMNS = ["start", "end", "taker", "taker_share"]


def merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """(start, end) runs, joining neighbours whose gap is at most `gap` frames."""
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def nearest_to_spot(players: np.ndarray) -> np.ndarray:
    """One team, per frame: distance (m) from the centre spot to its nearest tracked player (inf if none)."""
    d = np.linalg.norm(players - CENTRE, axis=2)
    return np.where(np.isnan(d), np.inf, d).min(axis=1)


def detect_kick_offs(tracking: atd.Tracking) -> pd.DataFrame:
    """One row per kick-off seen in the tracking: start, end (first and last frame of the layout; the ball is
    played ~1-2 s before `end`), taker ('H' / 'A', None if unclear), taker_share (frames the taker's team had
    the player nearest the spot, over TAKER_WINDOW_S)."""
    players, slot_side = tracking.players, tracking.slot_side
    rows = []
    for first, last in config.HALVES.values():
        lo = max(first - int(LEAD_S * FPS), 0)  # the opening kick-off is lined up before the graded half
        home, away = players[lo:last + 1, slot_side == "H"], players[lo:last + 1, slot_side == "A"]
        for s, e in merge_runs(kickoff_layouts(home, away), round(MERGE_GAP_S * FPS)):
            w_lo = max(s, e - round(TAKER_WINDOW_S[0] * FPS))
            w_hi = max(w_lo + 1, e - round(TAKER_WINDOW_S[1] * FPS))
            h_nearer = nearest_to_spot(home[w_lo:w_hi]) < nearest_to_spot(away[w_lo:w_hi])
            share_h = float(h_nearer.mean())
            taker, share = ("H", share_h) if share_h >= 0.5 else ("A", 1 - share_h)
            rows.append({"start": s + lo, "end": e + lo,
                         "taker": taker if share >= MIN_TAKER_SHARE else None, "taker_share": share})
    return pd.DataFrame(rows, columns=COLUMNS).astype({"start": int, "end": int})
