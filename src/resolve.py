"""Resolve: the one place where the detectors' answers are combined into labels.

Detectors never see each other. Any decision that depends on more than one source (one detector overruling
or confirming another) is written here, in order, as a commented line or a small helper. If a rule needs raw
data, load it through the cached loaders in `src.io`.

Trust order: tracking dead play > kick-off taker > touch possession > Smart Tagging. The tags still help
where the tracking is silent: set pieces are tagged reliably, and a tagged kick-off's restart frame is precise.
But a tagged kick-off is used only where the tracking sees the kick-off layout around it, with the tracking's
taker: the tags carry no taker for kick-offs, and one 'Kick Off (After Goal)' is really a drinks break.

The ball's own detector (`detectors.ball_out`: out moments, in play) then sharpens the dead ball's edges:
the tracking notices a stoppage once the players have slowed down (1-7 s late) and the tags start it when the
last tagged phase ended (up to 40 s early); the ball leaving the pitch is the moment the ground truth uses.
Where a dead segment and a ball-out event meet, the ball-out moment is the start (`ball_line_starts`); where
a tagged restart inside a dead segment is followed by the ball in play, the restart is the end
(`restart_ends`).
"""
import numpy as np
import pandas as pd

from . import config
from .detectors import bench_break, smart_tagging
from .detectors.touches import MAX_FLIP_GAP_S, first_flip
from .detectors.tracking_dead_play import runs_of, speed
from .io import atd

GOAL_END_TOL_S = 5.0  # a tracking dead segment starting this soon after the kick-off layout still ends the goal
KICK_OFFS = ("Kick Off (Start)", smart_tagging.KICK_OFF_AFTER_GOAL)
KICK_OFF_TOL_S = 5.0  # a tagged kick-off restart up to this long after its tracking layout broke still matches it
LONG_DEAD_S = 60.0  # a dead run this long can be a kick-off's wait (goals here: 64-255 s); the median stoppage is ~16 s
MAX_WAIT_GAP_S = 180.0  # never carry a kick-off's dead ball back over more non-dead frames than this

BALL_OUT_LOOKBACK_S = 10.0  # a stoppage starting this soon after the ball left the pitch started when it left
MAX_IN_PLAY_BEFORE = 0.2  # share of in-play frames tolerated between the ball leaving and the stoppage start
MIN_IN_PLAY_EARLY = 0.5  # share of in-play frames before the ball leaves that proves a tagged stoppage started early

RESTART_CHECK_S = 5.0  # the ball must be in play for RESTART_MIN_IN_PLAY of this long after a tagged restart...
RESTART_MIN_IN_PLAY = 0.5  # ...for the restart to end a stoppage that the tracking keeps running
RESTART_MIN_CUT_S = 0.5  # a restart closer than this to the stoppage's end changes nothing
RESTART_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Kick Off (Start)", smart_tagging.KICK_OFF_AFTER_GOAL)
# not "Free Kick": its tag starts at the foul, so its restart estimate is ~7 s before the kick is taken

BALL_RESTART_AFTER_S = 10.0  # the real kick or throw comes up to this long after the tags' restart estimate
BALL_RESTART_BEFORE_S = 3.0  # a stoppage ending this long before the tags' restart can still belong to it
MAX_IN_PLAY_GAP_S = 0.5  # the ball may be seen in play at most this long before the real restart (else it was live)

CORNER = "Corner Kick"
CORNER_LOOKBACK_S = 60.0  # a corner is taken within a minute of the ball crossing the goal line (here: 15-40 s)

KEEPER_IN_BOX_MIN = 0.8  # a stoppage spent mostly with the ball in view in a box is the keeper holding it
KEEPER_TOUCH_S = MAX_FLIP_GAP_S  # the box team's first touch comes this soon after the hidden ball is back
KEEPER_LIVE_AFTER_S = 5.0  # no stoppage this long after the hidden ball is back (else a goal kick, corner, foul)

BLIP_MAX_S = 3.0  # a possession shorter than this that runs straight into a stoppage is a touch without control

RESTART_LEAD_S = 1.0  # a touch or kick this long before a stoppage's end can be its restart (the edge is not exact)

GOAL_KICK = "Goal Kick"
SHOT_MAX_S = MAX_FLIP_GAP_S  # s: the shot's last touch is at most this long before the ball leaves (its flight)
SHOT_KICK_S = 0.4  # s: the ball's speed is read this long after the shot's touch...
SHOT_MIN_MS = 6.0  # m/s: ...and must pass this: kicked (`ball_out.RESTART_MS`), not dribbled or rolling
SHOT_MAX_WIDE_M = 40.3 / 2  # m from the goal's centre: a shot leaves within the penalty box's width
SHOT_OUT_MIN_S = 2.0  # s: a ball seen out this long left the pitch (`ball_out.OFF_MIN_S`), even with no stoppage yet


def ball_line_starts(
    dead: np.ndarray,
    tracked: np.ndarray,
    ball_out: pd.DataFrame,
    ball_in_play: np.ndarray,
) -> np.ndarray:
    """Start each dead segment when the ball crossed a line of the pitch, where the ball-out detector saw it.

    dead: the full dead mask (tracking + goals + tags). tracked: the part that came from tracking and goals.
    ball_out / ball_in_play: the ball-out detector's events and in-play mask.
    Two independent corrections per segment [s, e]:
     - noticed late: the ball left the pitch at o, up to BALL_OUT_LOOKBACK_S before s, and was not back in play
       between o and s (at most MAX_IN_PLAY_BEFORE of those frames): dead from o.
     - started early: the ball leaves the pitch at o inside the segment, it was in play before (at least
       MIN_IN_PLAY_EARLY of the frames) and that early part came from the tags only: live until o. Tracking
       starts are left alone: they are fouls, where the ball is kicked away after the whistle.
    """
    in_play = ball_in_play
    outs = ball_out.frame.to_numpy()
    back = round(BALL_OUT_LOOKBACK_S * config.FPS)
    out = dead.copy()
    for s, e in runs_of(dead):
        before = outs[(outs >= s - back) & (outs < s)]
        if len(before):
            o = int(before[-1])
            if not dead[o:s].any() and in_play[o:s].mean() <= MAX_IN_PLAY_BEFORE:
                out[o:s] = True
        within = outs[(outs > s) & (outs <= e)]
        if len(within):
            o = int(within[0])
            if not tracked[s:o].any() and in_play[s:o].mean() >= MIN_IN_PLAY_EARLY:
                out[s:o] = False
    return out


def restart_ends(
    dead: np.ndarray,
    set_pieces: pd.DataFrame,
    ball_out: pd.DataFrame,
    ball_in_play: np.ndarray,
) -> np.ndarray:
    """End a dead segment at the tagged restart inside it when the ball is in play right after that restart.

    The tracking ends a stoppage once the players speed up again, 0.5-2.5 s after the restart (up to 30 s
    when play resumes slowly). The tags' restart estimate is reliable for throw-ins, corners, goal kicks and
    kick-offs (RESTART_TYPES). Per segment: the last such restart at least RESTART_MIN_CUT_S before the end;
    if the ball is in play for RESTART_MIN_IN_PLAY of the RESTART_CHECK_S after it, the segment ends there.
    When the ball leaves the pitch again before the segment's end, the dead ball resumes from that moment
    (a kick-off followed by a quick goal kick).
    """
    in_play = ball_in_play
    outs = ball_out.frame.to_numpy()
    restarts = set_pieces.restart[set_pieces.type.isin(RESTART_TYPES)].to_numpy()
    check = round(RESTART_CHECK_S * config.FPS)
    out = dead.copy()
    for s, e in runs_of(dead):
        inside = restarts[(restarts >= s) & (restarts <= e + 1 - RESTART_MIN_CUT_S * config.FPS)]
        if not len(inside):
            continue
        r = int(inside[-1])
        if in_play[r:r + check].mean() < RESTART_MIN_IN_PLAY:
            continue
        again = outs[(outs > r) & (outs <= e)]
        out[r:int(again[0]) if len(again) else e + 1] = False
    return out


def ball_restart_ends(
    dead: np.ndarray,
    set_pieces: pd.DataFrame,
    ball_restarts: pd.DataFrame,
    ball_in_play: np.ndarray,
) -> np.ndarray:
    """Make a stoppage last until the kick or throw that restarts it, where the ball track shows it.

    The tags do not time the restart itself: their estimate (`smart_tagging.RESTART_LEAD`, set on kick-offs)
    is 2-4 s early for throw-ins and corners, and before a corner the players start their runs, so the
    tracking stops calling it dead too. Per tagged restart r (not kick-offs, whose tagged frame is precise and
    which the ball track sees ~0.2 s late): the dead segment ending closest to r, between
    BALL_RESTART_BEFORE_S before it and BALL_RESTART_AFTER_S after it (end e); the first moment after e and
    within BALL_RESTART_AFTER_S of r where the ball is set moving again (`ball_out.detect_ball_restarts`, k).
    If the ball is not in play for more than MAX_IN_PLAY_GAP_S between e and k, the dead ball runs on to k.
    Only ever extends: a stoppage that ends late is left to `restart_ends`.
    """
    out = dead.copy()
    runs = runs_of(dead)
    ends = np.array([e + 1 for _, e in runs])
    kicks = ball_restarts.frame.to_numpy()
    before, after = round(BALL_RESTART_BEFORE_S * config.FPS), round(BALL_RESTART_AFTER_S * config.FPS)
    gap = round(MAX_IN_PLAY_GAP_S * config.FPS)
    estimated = set_pieces[~set_pieces.type.isin(KICK_OFFS)]  # a tagged kick-off's restart frame is precise already
    for r in estimated.restart.astype(int):
        near = ends[(ends >= r - before) & (ends <= r + after)]
        if not len(near):
            continue
        e = int(near[np.argmin(np.abs(near - r))])
        later = kicks[(kicks > e) & (kicks <= r + after)]
        if len(later) and ball_in_play[e:int(later[0])].sum() <= gap:
            out[e:int(later[0])] = True
    return out


def corner_fills(
    dead: np.ndarray,
    set_pieces: pd.DataFrame,
    ball_out: pd.DataFrame,
    ball_lost: pd.DataFrame,
    ball_in_play: np.ndarray,
    corner_setups: pd.DataFrame,
) -> np.ndarray:
    """Dead from the ball crossing the goal line to each tagged corner kick, when the players set up for it.

    The tracking notices a corner's stoppage only once the players stand still in the box, 11-17 s after the
    ball went out; until then they walk into the box at about open-play pace. Per tagged corner (restart r):
    the last time the ball crossed a goal line within CORNER_LOOKBACK_S before r (o). Near the corners the track
    often loses the ball just before it crosses, so where no crossing is seen, the last time the ball was lost
    next to a goal line (`ball_out.detect_ball_lost_at_goal_line`) stands in for it. If a corner setup from the
    tracking (`corner_setup.detect_corner_setups`) overlaps [o, r) and the ball is in play for at most
    MAX_IN_PLAY_BEFORE of [o, r) (it really stayed out), the whole [o, r) is dead. Never past the kick, and
    never from a setup alone: a full box of walking players also happens in live play.
    """
    out = dead.copy()
    crossed_at = ball_out.frame[ball_out.edge == "goal line"].to_numpy()
    lost_at = ball_lost.frame.to_numpy()
    back = round(CORNER_LOOKBACK_S * config.FPS)
    for r in set_pieces.restart[set_pieces.type == CORNER].astype(int):
        crossed = crossed_at[(crossed_at < r) & (crossed_at >= r - back)]
        lost = lost_at[(lost_at < r) & (lost_at >= r - back)]
        if len(crossed):
            o = int(crossed[-1])
        elif len(lost):
            o = int(lost[-1])  # the ball vanished at the goal line: the best guess of when it went out
        else:
            continue
        set_up = ((corner_setups.start < r) & (corner_setups.end >= o)).any()
        if set_up and ball_in_play[o:r].mean() <= MAX_IN_PLAY_BEFORE:
            out[o:r] = True
    return out


def true_kick_offs(set_pieces: pd.DataFrame, kick_offs: pd.DataFrame) -> pd.DataFrame:
    """The tagged kick-offs the tracking confirms, as `set_pieces` rows (restart from the tag, the precise one).

    A tagged kick-off is true if its restart falls inside a tracking kick-off layout, or up to KICK_OFF_TOL_S
    after it breaks; the others are dropped (the drinks break tagged as a goal). Its taker is the tracking's,
    which reads it off the centre spot; the tag's inferred taker only where the tracking's is unclear.
    A tracking kick-off with no tag is not used: the tags miss none in this match.
    """
    tagged = set_pieces[set_pieces.type.isin(KICK_OFFS)]
    tol = round(KICK_OFF_TOL_S * config.FPS)
    rows = []
    for row in tagged.itertuples(index=False):
        hit = kick_offs[(kick_offs.start <= row.restart) & (row.restart <= kick_offs.end + tol)]
        if len(hit):
            taker = hit.taker.iloc[0]
            rows.append(row._replace(taker=taker if taker in ("H", "A") else row.taker))
    return pd.DataFrame(rows, columns=set_pieces.columns)


def kick_off_possession(kick_offs: pd.DataFrame, touches: pd.DataFrame, dead: np.ndarray | None = None) -> np.ndarray:
    """(N_FRAMES,) the taker from each kick-off (`true_kick_offs`) until the touches confirm the other team has the ball.

    The confirmation is the touch possession's own rule (`touches.first_flip`): MIN_FLIP consecutive touches by
    the other team, each within MAX_FLIP_GAP_S of the previous one, or two once a pass completes inside the run
    (where `touches` has `pass_in`), dated back to the first of them. Until then the taker holds the ball whatever the touches say, so a
    missed kick-off touch or a stray first touch cannot hand the opening possession to the wrong team.
    With the dead mask, the taker's hold also ends at the first stoppage after the kick-off: whoever restarts
    after it is no longer the kick-off's business (163937: H's goal kick after A's kick-off at 163085 stayed A).
    Kick-offs with an unknown taker are skipped. X elsewhere.
    """
    out = np.full(config.N_FRAMES, "X", dtype="<U1")
    kick_offs = kick_offs[kick_offs.taker.isin(("H", "A"))]
    for restart, taker in kick_offs[["restart", "taker"]].itertuples(index=False):
        half_first = next(lo for lo, hi in config.HALVES.values() if restart <= hi)
        start = max(int(restart), half_first)  # the first-half kick-off is tagged just before the graded half
        out[start:hold_end(start, taker, touches, dead)] = taker
    return out


def hold_end(start: int, team: str, touches: pd.DataFrame, dead: np.ndarray | None = None) -> int:
    """End (exclusive) of `team`'s hold on the ball from `start`, whatever the touch possession says meanwhile.

    The hold ends where the touches confirm the other team has the ball (`touches.first_flip`, dated back to the
    first touch of the run), or at the half's end. With the dead mask, also at the first stoppage after the
    hold's first live frame.
    """
    half_last = next(hi for lo, hi in config.HALVES.values() if start <= hi)
    later = touches[(touches.frame >= start) & (touches.frame <= half_last)]
    pass_in = later.pass_in.to_numpy() if "pass_in" in later else None
    weak = later.at_apex.to_numpy() if "at_apex" in later else None
    k = first_flip(later.frame.to_numpy(), (later.team != team).to_numpy(), pass_in=pass_in, weak=weak)
    end = int(later.frame.iloc[k]) if k >= 0 else half_last + 1
    if dead is not None:
        alive = start + int(np.argmax(~dead[start:end])) if (~dead[start:end]).any() else end
        stops = np.flatnonzero(dead[alive:end])
        end = alive + int(stops[0]) if len(stops) else end  # the first stoppage after the start ends it
    return end


def keeper_gathers(
    ball_hidden_in_box: pd.DataFrame, touches: pd.DataFrame, dead: np.ndarray, set_pieces: pd.DataFrame,
) -> list[tuple[int, int, str]]:
    """[start, end, team): the keeper's team has the ball from the moment the track lost it in their box.

    The keeper gathers the ball out of view (a dive, a catch among the players) and the track loses it until he
    plays it out; the touch possession only flips once the keeper's team has touched the ball MIN_FLIP times
    after that, 6-41 s late here (55514: the keeper's throw at 37:00). A ball lost in a box and found again at
    it (`keeper_hold.detect_ball_hidden_in_box`) is the keeper's team's from the frame it was lost when:
     - the first touch after the ball is back is by the box's team, within KEEPER_TOUCH_S: the keeper's kick or
       throw, from the box or landing with a teammate (the attackers keeping it in a crowded box, 89:17, fail);
     - nothing stops play from the loss to KEEPER_LIVE_AFTER_S after the ball is back: no dead frame after it
       and no tagged restart (a goal kick, a corner, a foul given in the box are not the keeper's ball in hand).
    The hold then runs as a kick-off's does (`hold_end`): until the touches confirm the other team has the ball,
    or the first stoppage. Dead frames inside it stay dead.
    """
    touch_len = round(KEEPER_TOUCH_S * config.FPS)
    live_len = round(KEEPER_LIVE_AFTER_S * config.FPS)
    restarts = set_pieces["restart"].to_numpy()
    spells = []
    for start, back, team in ball_hidden_in_box[["start", "back", "team"]].itertuples(index=False):
        later = touches[touches.frame >= back]
        if not len(later) or later.frame.iloc[0] > back + touch_len or later.team.iloc[0] != team:
            continue  # the box's team does not play the ball first once it is back
        if dead[back:back + live_len].any() or ((restarts >= start) & (restarts <= back + live_len)).any():
            continue  # play stops: a goal kick, a corner or a foul, not the keeper's ball
        spells.append((int(start), hold_end(int(start), team, touches, dead), team))
    return spells


def goal_dead_intervals(
    goal_celebrations: pd.DataFrame,
    detected_dead_play: np.ndarray,
    kick_offs: pd.DataFrame
) -> list[tuple[int, int]]:
    """[start, end) of the dead ball after each detected goal.

    Start: the detected goal frame, always (it beats the tags' kick-off stoppage and any tracking dead segment).
    End, by trust:
     1. the end of the tracking dead segment running when the teams line up (it contains the celebration end,
        or starts within GOAL_END_TOL_S after it): the players stand still until the kick-off is taken;
     2. else the first true kick-off restart (`true_kick_offs`) after the goal (before the next goal, same half);
     3. else the celebration end (the kick-off layout) from the goal detector.
    In every case the dead ball lasts at least until that true kick-off restart: the ball sits on the centre
    spot until it is played, and the players can start moving before the kick (75501 to 75542 after goal 2).
    Ending earlier would hand the gap to whatever possession ran before the goal, the scorers'.
    """
    segments = runs_of(detected_dead_play)
    goals = goal_celebrations.sort_values("goal_frame")
    next_goal = list(goals.goal_frame[1:]) + [config.N_FRAMES]
    kick_offs = kick_offs[kick_offs.type == smart_tagging.KICK_OFF_AFTER_GOAL]
    intervals = []
    for (goal, celebration_end), later in zip(goals[["goal_frame", "end_frame"]].itertuples(index=False), next_goal):
        half_last = next(last for first, last in config.HALVES.values() if first <= goal <= last)
        tol = round(GOAL_END_TOL_S * config.FPS)
        lined_up = [e + 1 for s, e in segments if s <= celebration_end + tol and e >= celebration_end]
        restarts = kick_offs.restart[(kick_offs.restart > goal) & (kick_offs.restart < min(later, half_last + 1))]
        if lined_up:
            end = lined_up[0]  # 1. tracking dead play
        elif len(restarts):
            end = int(restarts.iloc[0])  # 2. the tagged kick-off
        else:
            end = int(celebration_end)  # 3. the kick-off layout
        if len(restarts):
            end = max(end, int(restarts.iloc[0]))  # never alive before the true kick-off is taken
        intervals.append((int(goal), max(end, int(goal) + 1)))
    return intervals


def kick_off_waits(dead: np.ndarray, kick_offs: pd.DataFrame) -> list[tuple[int, int]]:
    """[start, end) from the end of the latest long dead run before each true kick-off (`true_kick_offs`) to its restart.

    The ball is dead from a goal or the half-time break until the kick-off, but players walking into position
    (and warm-up passes before the 2nd half) break that wait up with short false H / A runs. A long run is
    one of at least LONG_DEAD_S, or one starting at the half's first graded frame (the half-time break runs
    on before it). Only runs in the restart's own half count. A kick-off with no long run before it, or one
    that would turn more than MAX_WAIT_GAP_S of non-dead frames dead, is left alone: that long run is an
    older stoppage, not this kick-off's wait. A restart outside the graded halves gives nothing.
    """
    long_len = round(LONG_DEAD_S * config.FPS)
    max_gap = round(MAX_WAIT_GAP_S * config.FPS)
    waits = []
    for restart in kick_offs.restart.astype(int):
        half_first = next((first for first, last in config.HALVES.values() if first <= restart <= last), None)
        if half_first is None:
            continue  # the first-half kick-off is tagged just before the graded half
        long_ends = [e for s, e in runs_of(dead[half_first:restart]) if s == 0 or e - s + 1 >= long_len]
        if not long_ends:
            continue
        start = half_first + long_ends[-1] + 1
        if start < restart and restart - start <= max_gap:
            waits.append((start, restart))
    return waits


def keeper_holds_alive(
    detected_dead_play: np.ndarray, set_pieces: pd.DataFrame, ball_out: pd.DataFrame, ball_in_box: np.ndarray,
) -> np.ndarray:
    """detected_dead_play without the stretches where the keeper is holding the ball in his box.

    A tracking dead-play stretch with the ball in view inside a box (`keeper_hold.detect_ball_in_box`) for at
    least KEEPER_IN_BOX_MIN of it is play, not a stoppage, unless the ball left the pitch in the
    BALL_OUT_LOOKBACK_S before it or during it (a goal kick or a corner being set up), or the tags place a
    restart inside it (a goal kick the ball-out detector missed).
    """
    out = detected_dead_play.copy()
    lookback = round(BALL_OUT_LOOKBACK_S * config.FPS)
    out_frames = ball_out["frame"].to_numpy()
    restarts = set_pieces["restart"].to_numpy()
    for start, end in runs_of(detected_dead_play):
        if ball_in_box[start:end + 1].mean() < KEEPER_IN_BOX_MIN:
            continue
        if ((out_frames >= start - lookback) & (out_frames <= end)).any():
            continue
        if ((restarts >= start) & (restarts <= end)).any():
            continue
        out[start:end + 1] = False
    return out


def drop_stoppage_blips(labels: np.ndarray, touches: pd.DataFrame, set_pieces: pd.DataFrame) -> np.ndarray:
    """labels where a team's brief spell straight into a stoppage goes back to the team that had the ball.

    labels: H / A / D per frame (dead ball already applied). Pattern: team X has the ball, team Y gets a spell
    shorter than BLIP_MAX_S, then the ball is dead. That is Y knocking the ball out (a clearance, a block, a
    deflection), not Y gaining control, and "a simple touch of the ball without control" does not change
    possession (CHALLENGE.md). The spell goes back to X.
    Y's label carried on past the stoppage is no evidence of its own, so the next spell goes back to X too when
    it is Y's with no Y touch in it (157010: A touched the ball out, H's throw-in was labelled A until H's
    first touch), unless a tagged restart inside the stoppage names Y as its taker (42751: a corner by A, whose
    kick the touches missed). Never across half time.
    """
    out = labels.copy()
    max_len = round(BLIP_MAX_S * config.FPS)
    takers = set_pieces.dropna(subset=["taker"])
    for first, last in config.HALVES.values():
        window = labels[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        segs = [(first + a, first + b, window[a]) for a, b in zip(bounds[:-1], bounds[1:])]
        for i in range(1, len(segs) - 1):
            (_, _, before), (a, b, team), (_, _, after) = segs[i - 1], segs[i], segs[i + 1]
            if not (before in ("H", "A") and team in ("H", "A") and before != team and after == "D"):
                continue
            if b - a >= max_len:
                continue
            out[a:b] = before  # the touch into the stoppage was not control
            if i + 2 < len(segs):
                c, d, carried = segs[i + 2]
                own_touch = ((touches.frame >= c) & (touches.frame < d) & (touches.team == team)).any()
                own_restart = ((takers.restart >= b) & (takers.restart <= c) & (takers.taker == team)).any()
                if carried == team and not own_touch and not own_restart:
                    out[c:d] = before  # only carried across the stoppage from the blip
    return out


def restart_possession(labels: np.ndarray, touches: pd.DataFrame, ball_restarts: pd.DataFrame) -> np.ndarray:
    """labels where the team that restarts after a stoppage has the ball from the restart, not from its first touch.

    labels: H / A / D per frame. The touch possession holds the last team through a stoppage, so after it the
    old team X stays labelled until the restarting team Y's first detected touch: a false X spell, often under
    a second, before every restart X did not take. Pattern: a stoppage, then an X spell, then a Y spell. The X
    spell goes to Y when all of these hold:
     - X never touches the ball in it (from RESTART_LEAD_S before it starts): the label is only carried over;
     - Y's spell starts within MAX_FLIP_GAP_S of the restart (the longest flight of a kicked ball): Y's first
       touch is the receipt of the restart, not a regain after a long silence with the ball untracked;
     - no ball restart (`ball_out.detect_ball_restarts`) in it has an X player nearest the ball: X taking a
       corner that Y heads away at once, with the kick itself not detected as a touch (34830, 42751).
    Never across half time.
    """
    out = labels.copy()
    lead = round(RESTART_LEAD_S * config.FPS)
    max_gap = round(MAX_FLIP_GAP_S * config.FPS)
    for first, last in config.HALVES.values():
        window = labels[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        segs = [(first + a, first + b, window[a]) for a, b in zip(bounds[:-1], bounds[1:])]
        for i in range(1, len(segs) - 1):
            (_, _, before), (a, b, team), (_, _, after) = segs[i - 1], segs[i], segs[i + 1]
            if not (before == "D" and team in ("H", "A") and after in ("H", "A") and after != team):
                continue
            if b - a > max_gap:
                continue
            own_touch = ((touches.frame >= a - lead) & (touches.frame < b) & (touches.team == team)).any()
            kicks = ball_restarts[(ball_restarts.frame >= a - lead) & (ball_restarts.frame < b)]
            if not own_touch and not (kicks.team == team).any():
                out[a:b] = after  # Y restarted: the ball is Y's from the restart
    return out


def attacked_goal_lines() -> dict[tuple[int, str], float]:
    """Map (half, team) to the x of the goal line the team attacks: the end away from its players' median x in the half."""
    tracking = atd.load()
    out = {}
    for half, (first, last) in config.HALVES.items():
        for team in ("H", "A"):
            x = np.nanmedian(tracking.players[first:last + 1, tracking.slot_side == team, 0])
            out[(half, team)] = config.PITCH_LENGTH if x < config.PITCH_LENGTH / 2 else 0.0
    return out


def missed_shots(
    labels: np.ndarray,
    dead: np.ndarray,
    touches: pd.DataFrame,
    ball_out: pd.DataFrame,
    ball_lost: pd.DataFrame,
    set_pieces: pd.DataFrame,
) -> list[tuple[int, int, str]]:
    """[start, end, team): the spells of a team whose shot went wide over the goal line it attacks.

    Team Y wins the ball and shoots at once, with one or two touches: too few for the touch possession to flip,
    and `drop_stoppage_blips` reads a touch straight into a stoppage as a clearance. But a shot is an attacking
    action. The exit o is any of:
     - the ball seen crossing a goal line (`ball_out`, edge 'goal line'), with a dead frame within
       BALL_OUT_LOOKBACK_S after it or the ball staying out for at least SHOT_OUT_MIN_S (65254: the stoppage is
       noticed 13 s later); Y is the team attacking that end (`attacked_goal_lines`);
     - the ball lost by the track next to a goal line (`ball_lost`), with a dead frame within BALL_OUT_LOOKBACK_S;
     - a tagged goal kick by X whose stoppage is in the dead mask: o is the start of that dead run, Y the other
       team; the tracking can notice the stoppage late, so the touch may be up to BALL_OUT_LOOKBACK_S before o,
       and it must be in the half Y attacks (the tag's taker agrees with the pitch).
    The last touch before o must be Y's, at most SHOT_MAX_S before it (the shot's flight), in live play, and a
    shot: the ball leaves it faster than SHOT_MIN_MS, and a line exit is within SHOT_MAX_WIDE_M of the goal's
    centre (34157, 137582: a dribble lost at the corner flag is not one). Y's spell runs from the first touch of
    Y's run up to it (consecutive Y touches, each within MAX_FLIP_GAP_S, none across a dead frame) to o. Spells
    already labelled Y on their live frames are skipped. Never across half time.
    """
    attacked = attacked_goal_lines()
    tracking = atd.load()
    ball_speed = speed(tracking.ball)
    back = round(BALL_OUT_LOOKBACK_S * config.FPS)
    stays_out = round(SHOT_OUT_MIN_S * config.FPS)
    half_of = lambda f: next((h for h, (first, last) in config.HALVES.items() if first <= f <= last), None)
    exits = []  # (exit frame, shooter, longest touch-to-exit gap in frames, y where the ball left or NaN)
    lines = pd.concat([ball_out[ball_out.edge == "goal line"].assign(out_for=lambda d: np.where(
                           d.back < 0, config.N_FRAMES, d.back) - d.frame), ball_lost.assign(out_for=0)])
    for o, x, y, out_for in lines[["frame", "x", "y", "out_for"]].itertuples(index=False):
        o = int(o)
        if half_of(o) is None or not (dead[o:o + back].any() or out_for >= stays_out):
            continue
        end = 0.0 if x < config.PITCH_LENGTH / 2 else config.PITCH_LENGTH
        shooter = next(team for team in ("H", "A") if attacked[(half_of(o), team)] == end)
        exits.append((o, shooter, round(SHOT_MAX_S * config.FPS), y))
    for r, taker in set_pieces.loc[set_pieces.type == GOAL_KICK, ["restart", "taker"]].itertuples(index=False):
        r = int(r)
        if taker not in ("H", "A") or half_of(r - 1) is None or not dead[r - 1]:
            continue
        runs = [s for s, e in runs_of(dead[:r]) if e == r - 1]  # the dead run the goal kick ends
        exits.append((int(runs[0]), "A" if taker == "H" else "H", back, np.nan))

    frames, teams = touches.frame.to_numpy(), touches.team.to_numpy()
    max_gap = MAX_FLIP_GAP_S * config.FPS
    kick = round(SHOT_KICK_S * config.FPS)
    spells = {}  # start: (end, team)
    for o, team, gap, y in exits:
        half = half_of(o)
        half_first = config.HALVES[half][0]
        i = int(np.searchsorted(frames, o)) - 1  # the last touch before the exit: the shot
        if i < 0 or frames[i] < half_first or teams[i] != team or o - frames[i] > gap or dead[frames[i]]:
            continue
        shot = ball_speed[frames[i]:frames[i] + kick]
        if not (np.isfinite(shot).any() and np.nanmax(shot) > SHOT_MIN_MS):
            continue  # not kicked: a dribble or a ball rolling out
        if np.isfinite(y) and abs(y - config.PITCH_WIDTH / 2) > SHOT_MAX_WIDE_M:
            continue  # out far from the goal: not aimed at it
        if np.isnan(y):  # goal kick: the shot came from the half the shooter attacks
            bx = tracking.ball[frames[i], 0]
            if np.isnan(bx) or abs(bx - attacked[(half, team)]) > config.PITCH_LENGTH / 2:
                continue
        j = i
        while (j > 0 and teams[j - 1] == team and frames[j] - frames[j - 1] <= max_gap and frames[j - 1] >= half_first
               and not dead[frames[j - 1]:frames[j]].any()):
            j -= 1
        start = int(frames[j])
        if (labels[start:o][~dead[start:o]] == team).all():
            continue  # the shooters had the ball already
        spells[start] = max(spells.get(start, (o, team)), (o, team))  # the same shot seen by several exits
    return [(start, end, team) for start, (end, team) in sorted(spells.items())]


def dead_mask(
    tags: pd.DataFrame,
    detected_dead_play: np.ndarray,
    goal_celebrations: pd.DataFrame,
    bench_breaks: pd.DataFrame,
    kick_offs: pd.DataFrame,
    ball_out: pd.DataFrame,
    ball_in_play: np.ndarray,
    ball_lost: pd.DataFrame,
    ball_restarts: pd.DataFrame,
    corner_setups: pd.DataFrame,
    ball_in_box: np.ndarray,
    ball_off_pitch: pd.DataFrame,
    ball_line: bool = True,
    restart_end: bool = True,
    corner_fill: bool = True,
    ball_restart: bool = True,
    keeper_hold: bool = True,
    off_pitch: bool = True,
) -> np.ndarray:
    """Tracking dead play, the dead ball after each detected goal and at each bench break, then every set
    piece's stoppage they all missed.

    After a goal the detected goal frame is the start of the dead ball: a tracking dead segment already running
    at that frame is cut there (the frames before it are the scoring move). A bench break (walk to the benches,
    stand, walk back) is dead throughout, adding the walks the slow-players test misses. A set piece whose
    stoppage (last open play to restart) overlaps a tracking, goal or bench dead frame is left to them, which
    are more precise; one that overlaps none is taken from the tags. So the tags' kick-off stoppage after a goal
    never applies. Then each true kick-off's wait (`kick_off_waits`) is dead from the latest long dead run
    before it to its restart. kick_offs: the tracking kick-offs (`kick_off.detect_kick_offs`); a goal ends at
    a true one.

    Then the corrections from the ball's own track, each behind its switch, in this order:
     - keeper_hold: the keeper holding the ball is not a stoppage (`keeper_holds_alive`, before anything else);
     - corner_fill: dead from the ball crossing the goal line to a tagged corner kick (`corner_fills`);
     - ball_line: a stoppage starts where the ball crossed a line (`ball_line_starts`);
     - restart_end: a stoppage ends at the tagged restart when the ball is in play right after (`restart_ends`);
     - ball_restart: a stoppage runs on to the kick or throw seen in the ball track (`ball_restart_ends`);
     - off_pitch: every spell of the ball clearly off the pitch is dead (`ball_out.detect_ball_off_pitch`).
    """
    set_pieces = smart_tagging.set_pieces(tags)
    if keeper_hold:  # first: the steps below lean on the tracking stoppages
        detected_dead_play = keeper_holds_alive(detected_dead_play, set_pieces, ball_out, ball_in_box)
    dead = detected_dead_play.copy()
    confirmed = true_kick_offs(set_pieces, kick_offs)
    for start, end in goal_dead_intervals(goal_celebrations, detected_dead_play, confirmed):
        for s, e in runs_of(detected_dead_play):
            if s < start <= e:
                dead[s:start] = False  # the goal start beats the tracking segment's start
        dead[start:end] = True
    dead |= bench_break.bench_mask(bench_breaks)  # the walks to and from the benches are dead too
    tracked = dead.copy()  # tracking + goals + bench breaks, before the tags fill in
    for start, restart in set_pieces[["dead_start", "restart"]].itertuples(index=False):
        if not tracked[start:restart].any():
            dead[start:restart] = True
    for start, restart in kick_off_waits(dead, confirmed):
        dead[start:restart] = True  # the wait before a kick-off is dead back to its long stoppage
    if corner_fill:
        dead = corner_fills(dead, set_pieces, ball_out, ball_lost, ball_in_play, corner_setups)
    if ball_line:
        dead = ball_line_starts(dead, tracked, ball_out, ball_in_play)
    if restart_end:
        dead = restart_ends(dead, set_pieces, ball_out, ball_in_play)
    if ball_restart:
        dead = ball_restart_ends(dead, set_pieces, ball_restarts, ball_in_play)
    if off_pitch:
        for start, end in ball_off_pitch[["start", "end"]].itertuples(index=False):
            dead[start:end] = True
    return dead


def resolve(
    tags: pd.DataFrame,
    detected_dead_play: np.ndarray,
    touches: pd.DataFrame,
    touch_possession: np.ndarray,
    goal_celebrations: pd.DataFrame,
    bench_breaks: pd.DataFrame,
    kick_offs: pd.DataFrame,
    ball_out: pd.DataFrame,
    ball_in_play: np.ndarray,
    ball_lost: pd.DataFrame,
    ball_restarts: pd.DataFrame,
    corner_setups: pd.DataFrame,
    ball_in_box: np.ndarray,
    ball_off_pitch: pd.DataFrame,
    ball_hidden_in_box: pd.DataFrame,
    ball_line: bool = True,
    restart_end: bool = True,
    corner_fill: bool = True,
    ball_restart: bool = True,
    keeper_hold: bool = True,
    off_pitch: bool = True,
    stoppage_blips: bool = True,
    restart_team: bool = True,
    keeper_gather: bool = True,
    missed_shot: bool = True,
) -> np.ndarray:
    """Labels ('H' / 'A' / 'D' / 'X') per video frame. D wins over H / A; X where nothing is known.

    tags: the raw Smart Tagging table.
    detected_dead_play: bool dead-ball mask from the tracking.
    touches: one row per detected touch (frame, slot, team, dist_m, dv_ms, pass_in); ends each kick-off's possession.
    touch_possession: '<U1' H / A / X from the touches.
    goal_celebrations: one row per goal detected from the tracking (team, goal_frame, ..., end_frame).
    bench_breaks: one row per break at the benches from the tracking (touchline_y, start, ..., end).
    kick_offs: one row per kick-off seen in the tracking (start, end of the layout, taker, taker_share).
    ball_out: one row per time the ball left the pitch (frame, edge, x, y, back), from the ball-out detector.
    ball_in_play: bool mask, the ball inside the pitch and moving, from the ball-out detector.
    ball_lost: one row per time the track lost the ball next to a goal line (frame, x, y, seconds), ball-out detector.
    ball_restarts: one row per moment the ball is set moving after being still (frame, x, y, speed, team and dist_m
        of the nearest player), ball-out detector.
    corner_setups: one row per many-players-walking-in-a-box stretch (start, end, side, in_box), from the tracking.
    ball_in_box: bool mask, the ball tracked inside a penalty box, from the keeper-hold detector.
    ball_off_pitch: one row per spell of the ball clearly off the pitch (start, end, edge, deepest), ball-out detector.
    ball_hidden_in_box: one row per time the track lost the ball in a box and found it again at that box (start,
        back, team defending it, ...), from the keeper-hold detector.
    The switches (all on by default; off reproduces the output without that correction): see `dead_mask`;
    stoppage_blips: see `drop_stoppage_blips`; restart_team: see `restart_possession`; keeper_gather: see
    `keeper_gathers`; missed_shot: see `missed_shots`.
    """
    dead = dead_mask(
        tags, detected_dead_play, goal_celebrations, bench_breaks, kick_offs,
        ball_out, ball_in_play, ball_lost, ball_restarts, corner_setups, ball_in_box, ball_off_pitch, ball_line,
        restart_end, corner_fill, ball_restart, keeper_hold, off_pitch,
    )
    tag_possession = smart_tagging.possession(tags)
    labels = np.where(touch_possession != "X", touch_possession, tag_possession)  # touches first, tags fill gaps
    if stoppage_blips:  # needs the stoppages; before the kick-offs, so a kick-off taker is never reverted
        labels = drop_stoppage_blips(np.where(dead, "D", labels), touches, smart_tagging.set_pieces(tags))
    if restart_team:  # after the blips (they can change who the label carries), before the kick-offs (exact takers)
        labels = restart_possession(np.where(dead, "D", labels), touches, ball_restarts)
    confirmed = true_kick_offs(smart_tagging.set_pieces(tags), kick_offs)  # tagged kick-offs the tracking sees
    kick_off = kick_off_possession(confirmed, touches, dead if restart_team else None)
    labels = np.where(kick_off != "X", kick_off, labels)  # the kick-off taker beats both until touches confirm
    if keeper_gather:  # after the kick-offs: the keeper gathering the ball ends the taker's hold (76452)
        for start, end, team in keeper_gathers(ball_hidden_in_box, touches, dead, smart_tagging.set_pieces(tags)):
            labels[start:end] = team
    if missed_shot:  # last: a shot that went wide is the shooters' possession, whatever the blips or kick-off said
        for start, end, team in missed_shots(labels, dead, touches, ball_out, ball_lost, smart_tagging.set_pieces(tags)):
            labels[start:end] = team
    labels[dead] = "D"
    return labels
