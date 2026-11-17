"""Dead ball from the tracking alone: physics-based, no training, no tags.

A frame is dead when the players are slow *relative to how fast they move in this stretch of the match*
(median tracked-player speed below REL_V x its rolling median over BASE_S) for at least MIN_DEAD_S, AND that is
explained by either of two independent cases:
 a. the ball is also still (kick-off, ball in hand): ball speed below BALL_V_MAX for half the stretch; or
 b. the players decelerated into it (ball shot out, foul, whistle): their median speed fell by >= DECEL_FRAC of
    the baseline within the last LAG_S seconds. The ball may still be moving, so its speed is ignored.
Slow players alone, with a moving ball and no deceleration, is slow build-up play and is rejected.

The speed test fires ~1 s after the real event, so each segment is then pulled back to where the speed change
began (see `refine_boundaries`). Each half is detected separately (the baseline must not straddle half time).
All thresholds are physical priors, not tuned on the public ground truth.

Developed in dev/analysis/extract_dead_ball.py, which has the reasoning and the checks.
"""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..io import atd

FPS = config.FPS

WIN_S = 1.0  # smoothing window for speeds (rolling median)
BALL_V_MAX = 2.0  # m/s: the ball is "not moving" below this
REL_V = 0.75  # players are "slow" below this fraction of the rolling baseline
BASE_S = 600.0  # rolling-median window (10 min) that defines the baseline
MIN_DEAD_S = 5.0  # shortest stoppage reported
LAG_S = 3.0  # look-back used to measure the players' deceleration
DECEL_FRAC = 0.3  # speed drop (from the look-back peak to now), as a fraction of the baseline, that counts as 'players stopped'
MAX_GAP_S = 1.0  # gaps shorter than this inside a dead segment are bridged
SHIFT_FALLBACK_S = 1.0  # constant shift (both boundaries earlier) when a boundary has no clear onset
ONSET_TOL = 0.1  # fraction of the speed range counted as "still at the minimum" when locating the acceleration onset
MIN_CHANGE_FRAC = 0.05  # speed change (fraction of the baseline) below which there is no clear onset
SPEED_K = 3  # frames either side for the central-difference speed (0.12 s)
PAD = int(5 * FPS)  # context either side of each half, so its edge frames have a smoothing neighbourhood


def speed(xy: np.ndarray, k: int = SPEED_K) -> np.ndarray:
    """Central-difference speed in m/s, NaN where either end is missing. Works on (n, 2) or (n, slots, 2)."""
    v = np.full(xy.shape[:-1], np.nan)
    v[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * FPS / (2 * k)
    return v


def rolling_median(x: np.ndarray, seconds: float = WIN_S) -> np.ndarray:
    w = max(int(round(seconds * FPS)) | 1, 1)  # odd
    return pd.Series(x).rolling(w, center=True, min_periods=max(w // 4, 1)).median().to_numpy()


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of the True runs."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(s, e - 1) for s, e in zip(edges[::2], edges[1::2])]


def bridge_and_filter(mask: np.ndarray, min_len: int, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than max_gap, then drop True runs shorter than min_len (all in frames)."""
    def runs(m):
        edges = np.flatnonzero(np.diff(np.r_[False, m, False].astype(np.int8)))
        return list(zip(edges[::2], edges[1::2]))  # [start, end)
    out = mask.copy()
    for s, e in runs(~mask):
        if e - s < max_gap and s > 0 and e < len(mask):
            out[s:e] = True
    for s, e in runs(out):
        if e - s < min_len:
            out[s:e] = False
    return out


def refine_boundaries(dead: np.ndarray, player_speed: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Pull every segment's boundaries back to where the speed change began.

    Start: argmax of player_speed in the LAG_S before the detected start (deceleration onset).
    End:   last frame in the LAG_S before the detected end where the speed is still within ONSET_TOL of the
             window minimum (acceleration onset). Either falls back to a constant SHIFT_FALLBACK_S shift if it
             has no clear onset. Boundaries never cross the neighbouring segments; a segment keeps >= 1 frame.
    """
    lag, shift = int(LAG_S * FPS), int(SHIFT_FALLBACK_S * FPS)
    segs = runs_of(dead)
    out = np.zeros_like(dead)
    prev_end = -1
    for i, (s, e) in enumerate(segs):
        next_start = segs[i + 1][0] if i + 1 < len(segs) else len(dead)
        base = np.nanmean(baseline[s:e + 1])
        min_change = MIN_CHANGE_FRAC * base if np.isfinite(base) else np.inf

        new_s, new_e = s - shift, e - shift  # fallback
        lo = max(s - lag, 0)
        w = player_speed[lo:s + 1]
        if np.isfinite(w).any() and np.nanmax(w) - player_speed[s] >= min_change:
            new_s = lo + int(np.nanargmax(w))
        lo = max(e - lag, s)
        w = player_speed[lo:e + 1]
        if np.isfinite(w).any() and w[-1] - np.nanmin(w) >= min_change:
            near_min = np.flatnonzero(w <= np.nanmin(w) + ONSET_TOL * (w[-1] - np.nanmin(w)))
            new_e = lo + int(near_min[-1])

        new_s = max(new_s, prev_end + 1, 0)
        new_e = min(new_e, next_start - 1, len(dead) - 1)
        if new_e < new_s:  # degenerate: keep the original segment untouched
            new_s, new_e = max(s, prev_end + 1), e
        out[new_s:new_e + 1] = True
        prev_end = new_e
    return out


def detect_half(ball_speed: np.ndarray, player_speed: np.ndarray) -> np.ndarray:
    """One half (+ padding) to a bool mask; see the module docstring for the rule."""
    w = int(BASE_S * FPS) | 1
    baseline = pd.Series(player_speed).rolling(w, center=True, min_periods=w // 4).median().to_numpy()
    players_slow = np.where(np.isnan(player_speed) | np.isnan(baseline), False, player_speed < REL_V * baseline)
    slow = bridge_and_filter(players_slow, int(MIN_DEAD_S * FPS), int(MAX_GAP_S * FPS))
    ball_still = np.where(np.isnan(ball_speed), True, ball_speed < BALL_V_MAX)
    lag = int(LAG_S * FPS)
    dead = np.zeros_like(slow)
    for s, e in runs_of(slow):
        e += 1  # exclusive
        before = player_speed[max(s - lag, 0):s + 1]
        drop = np.nanmax(before) - np.nanmin(player_speed[s:min(s + int(FPS), e)]) if np.isfinite(before).any() else 0.0
        if ball_still[s:e].mean() >= 0.5 or drop >= DECEL_FRAC * np.nanmean(baseline[s:e]):
            dead[s:e] = True
    return refine_boundaries(dead, player_speed, baseline)


def detect_dead_play(tracking: atd.Tracking) -> np.ndarray:
    """(N_FRAMES,) bool: True where the tracking says the ball is dead. Only the graded halves are detected."""
    ball_speed = rolling_median(speed(tracking.ball))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where no slot is tracked
        player_speed = rolling_median(np.nanmedian(speed(tracking.players), axis=1))

    out = np.zeros(config.N_FRAMES, dtype=bool)
    for first, last in config.HALVES.values():
        lo, hi = first - PAD, last + PAD + 1
        detected = detect_half(ball_speed[lo:hi], player_speed[lo:hi])
        out[first:last + 1] = detected[PAD:PAD + last - first + 1]  # the padding is context only
    return out
