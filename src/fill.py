"""Gap filling: decide the frames the resolver left as 'X'."""
import numpy as np

from . import config


def carry_forward(labels: np.ndarray) -> np.ndarray:
    """Give each 'X' frame the last possession label ('H' / 'A') before it, within the same half.

    'D' frames are kept and never carried, so a stoppage does not leak into the live play after it.
    Never carries across half time. Frames before the first possession label of a half stay 'X'.
    """
    out = labels.copy()
    for first, last in config.HALVES.values():
        seg = out[first:last + 1]  # view into out
        idx = np.where(np.isin(seg, ("H", "A")), np.arange(len(seg)), -1)
        np.maximum.accumulate(idx, out=idx)  # index of the last possession frame so far
        fill = (seg == "X") & (idx >= 0)
        seg[fill] = seg[idx[fill]]
    return out
