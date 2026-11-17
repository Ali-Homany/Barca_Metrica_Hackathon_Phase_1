"""Detectors: one module per evidence source.

A detector works on the whole match and returns a full answer, one value per video frame (or a table of
events), from the data it is given:
- possession: (N_FRAMES,) '<U1' with 'H' / 'A' where the source has an opinion, else 'X'
- dead ball: (N_FRAMES,) bool, True where the ball is dead

The pipeline loads the data and calls the detectors; they do not load anything, do not call each other, and
do not decide between conflicting sources. That is `src/resolve.py`, the only place where answers are combined.
"""
