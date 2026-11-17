"""Possession and dead balls from Smart Tagging.

Every code and every set-piece Type in the file is listed below with its meaning; an unknown one raises,
so a changed export cannot be silently mislabelled.

Two detectors, both taking the table from `io.smart_tagging.load()`:
- `possession`: H/A from every phase that implies who has the ball (X elsewhere). Phases overlap and
  sometimes disagree; the most recently started one wins, since a new phase is the freshest sign of a change.
- `dead_tags`: True on the stoppage before every set piece, including half-time and post-goal kick-offs.
"""
import numpy as np
import pandas as pd

from .. import config

# What an interval of each code says, i.e. how to read its Team tag.
HAS_BALL = "has_ball"  # an open-play phase; Team is the team with the ball
DEFENDING = "defending"  # an open-play phase; Team is the team without the ball
SET_PIECE = "set_piece"  # a restart; Team is the taker, or N/A (see SET_PIECE_TYPES and `set_pieces`)
GOAL = "goal"  # Team scored; runs from the scoring move to just after the kick-off that follows,
#                so it mixes the move (Team's ball) with the dead ball after the goal
CONTEXT = "context"  # where the ball / players are, or a single action: not used. Its Team does not
#                      say who has the ball, and the zone codes run straight through stoppages
#                      (PLAYERS IN THE BOX is still running at 34 of the 59 restarts), so their ends
#                      do not mark play stopping either

CODES = {
    "BUILD UP": HAS_BALL,
    "PROGRESSION": HAS_BALL,
    "CREATING CHANCES": HAS_BALL,
    "ATTACKING TRANSITION": HAS_BALL,
    "DEFENDING IN DEFENSIVE THIRD": DEFENDING,
    "DEFENDING IN MIDDLE THIRD": DEFENDING,
    "DEFENDING IN ATTACKING THIRD": DEFENDING,
    "DEFENSIVE TRANSITION": DEFENDING,
    "SET PIECES": SET_PIECE,
    "GOALS": GOAL,
    "BALL IN FINAL THIRD": CONTEXT,
    "BALL IN THE BOX": CONTEXT,
    "PLAYERS IN THE BOX": CONTEXT,
    "LONG BALL": CONTEXT,  # one action; its interval covers the flight and the contest for the ball
}
OPEN_PLAY = (HAS_BALL, DEFENDING)  # phases that say who has the ball and that play is live

# Every set-piece Type, mapped to the longest stoppage (s) assumed before it; None = no cap, because kick-offs
# wait out half time or a goal celebration. A physical prior, not tuned.
SET_PIECE_TYPES = {
    "Throw-in": 60.0,
    "Corner Kick": 60.0,
    "Goal Kick": 60.0,
    "Free Kick": 60.0,
    "Kick Off (Start)": None,
    "Kick Off (After Goal)": None,
}
KICK_OFF_AFTER_GOAL = "Kick Off (After Goal)"  # taken by the team that conceded

# Frames from the start of a set-piece window to the restart itself. The two published kick-off frames
# (DATA.md: 11185 and 113742) fall 88 and 82 frames into their windows; assumed the same for every Type.
RESTART_LEAD = 85
MIN_DEAD_S = 3.0  # shortest stoppage before a restart (the tags often overlap the restart by ~1 s)

OPPONENT = {"H": "A", "A": "H"}


def annotate(t: pd.DataFrame) -> pd.DataFrame:
    """The Smart Tagging table plus a `role` column, after checking every code and Type is known."""
    if unknown := set(t.code) - CODES.keys():
        raise ValueError(f"unknown code(s) {unknown}: decide their role in CODES")
    if unknown := set(t.type[t.code.map(CODES) == SET_PIECE]) - SET_PIECE_TYPES.keys():
        raise ValueError(f"unknown set-piece Type(s) {unknown}: add them to SET_PIECE_TYPES")
    return t.assign(role=t.code.map(CODES))


def possessor(role: str, team) -> str | None:
    """Side with the ball implied by an open-play or goal interval, or None if it does not say."""
    if team not in OPPONENT:  # N/A arrives as NaN from pandas, not None
        return None
    if role in (HAS_BALL, GOAL):
        return team
    if role == DEFENDING:
        return OPPONENT[team]
    return None  # CONTEXT, and SET_PIECE (its taker is resolved in `set_pieces`)


def set_pieces(tagging: pd.DataFrame) -> pd.DataFrame:
    """One row per set piece: `restart` frame, `taker` side (None if unknown), first dead frame `dead_start`.

    The stoppage runs from the end of the last open-play phase that finished before the restart, bounded
    by the start of the half, the previous restart, the Type's cap and MIN_DEAD_S.
    The taker is the Team tag; for N/A, the team that conceded (kick-off after a goal) or the team of the
    first open-play phase starting inside the set-piece window.
    """
    t = annotate(tagging)
    play = t[t.role.isin(OPEN_PLAY)]
    goals = t[t.role == GOAL]
    rows, prev_restart = [], 0
    for sp in t[t.role == SET_PIECE].itertuples():
        restart = sp.start_frame + RESTART_LEAD

        half_first = max((first for first, _ in config.HALVES.values() if first <= restart), default=0)
        ended = play.end_frame[(play.start_frame < restart) & (play.end_frame <= restart)]
        dead_start = max(ended.max() if len(ended) else 0, half_first, prev_restart)
        if (cap := SET_PIECE_TYPES[sp.type]) is not None:
            dead_start = max(dead_start, restart - round(cap * config.FPS))
        dead_start = min(dead_start, restart - round(MIN_DEAD_S * config.FPS))

        taker = sp.team if sp.team in OPPONENT else None
        if taker is None and sp.type == KICK_OFF_AFTER_GOAL:
            scored = goals[goals.start_frame <= sp.start_frame]
            taker = OPPONENT.get(scored.team.iloc[-1]) if len(scored) else None
        elif taker is None:
            nxt = play[(play.start_frame >= restart) & (play.start_frame < sp.end_frame)]
            taker = possessor(nxt.role.iloc[0], nxt.team.iloc[0]) if len(nxt) else None

        rows.append({"type": sp.type, "restart": restart, "end_frame": sp.end_frame,
                     "taker": taker, "dead_start": dead_start})
        prev_restart = restart
    return pd.DataFrame(rows)


def possession(tagging: pd.DataFrame) -> np.ndarray:
    """(N_FRAMES,) H/A over every interval that implies who has the ball, later-starting intervals on top; X elsewhere.

    Open-play and goal intervals count from their start; set pieces from their restart, for the taker.
    """
    t = annotate(tagging)
    play = t[t.role.isin((*OPEN_PLAY, GOAL))]
    sp = set_pieces(tagging)
    spans = pd.concat([
        pd.DataFrame({"start": play.start_frame, "end": play.end_frame,
                      "side": [possessor(r, team) for r, team in zip(play.role, play.team)]}),
        pd.DataFrame({"start": sp.restart, "end": sp.end_frame, "side": sp.taker}),
    ]).sort_values("start", kind="stable")

    out = np.full(config.N_FRAMES, "X", dtype="<U1")
    for start, end, side in spans.itertuples(index=False):
        if side in OPPONENT:
            out[start:end] = side
    return out


def dead_tags(tagging: pd.DataFrame) -> np.ndarray:
    """(N_FRAMES,) bool: True on the stoppage before every set piece (including half-time and post-goal kick-offs)."""
    out = np.zeros(config.N_FRAMES, dtype=bool)
    for start, restart in set_pieces(tagging)[["dead_start", "restart"]].itertuples(index=False):
        out[start:restart] = True
    return out
