# TikiData - More than a Hack 2027, Phase 1

Technical document. Ball possession for every frame of BAR-ATH.

## 1. Approach

We label every frame with hand-written rules on top of the tracking data (ATD) and the Smart Tagging. No model is trained. For each frame the pipeline answers two questions: is the ball dead, and if not, which team has it.

Why rules and not a trained model: the only possession labels for this match are the public five minutes, which may not be used for training or tuning, and we had no labelled possession data from other games. The ground truth is derived from corrected tracking, so most of the work is reading the automatic tracking well: finding touches, stoppages and restarts. These are physical events (the ball changes direction next to a player, the players stop running, the ball leaves the pitch) and can be written down directly, with thresholds in metres and seconds. Rules are also easy to check one by one, and every correction can be switched off to see what it does.

The code has three parts. Each detector in `src/detectors/` reads the data and gives one answer per frame or a list of events, without looking at the other detectors. `src/resolve.py` is the only place where their answers are combined, in a fixed order. `src/fill.py` fills the frames that are still open. When sources disagree the order of trust is: dead ball from the tracking, then the kick-off taker, then possession from touches, then the Smart Tagging.

## 2. Assets

Used:

- Tactical ATD (txt and xml): ball and player positions per frame, and the team of each player slot. This is the main source for everything.
- Tactical Smart Tagging (xml): set pieces (type, taker, rough restart time) and open-play phases (which team is building up, defending, and so on). Used where the tracking says nothing, and to know which kind of restart a stoppage ends with.
- Pitch size 105.3 x 68 m, from the organisers' `games.py`.

Not used:

- The tactical video. We watched it to understand the data and debug the pipeline, but nothing in the pipeline reads it.
- The homography: the ATD is already in pitch coordinates, so we did not need it.
- The broadcast video and its tracking: they are not on the same timeline as the tactical video.

## 3. Training data

None. We used no labelled dataset and trained no model. The thresholds are set from physical reasoning (player and ball speeds, contact distances, penalty box size from the Laws of the Game). The only outside data is the Metrica Sports open sample data, games 1 to 3 (https://github.com/metrica-sports/sample-data), used to check our thresholds on other matches. Nothing was fitted on it.

The public five minutes of ground truth were used only to check the pipeline end to end (frame alignment, file format, labels meaning what we think).

## 4. Possession rules

### 4.1 Detectors

Each detector reads the ATD or the tags and reports what it sees, on its own. None of them decides a label by itself; that happens in 4.2 and 4.3.

- Dead play (`tracking_dead_play.py`): the players are much slower than usual (median speed below 75% of its 10-minute rolling median) for at least 5 s, with the ball still or a sharp slowdown just before. The edges are moved back to where the slowdown started.
- Ball out (`ball_out.py`): when the ball crosses a line; when it is in play (inside the pitch and faster than 3 m/s); when the track loses it next to a goal line; when it is kicked or thrown after being still (faster than 6 m/s after 1 s still); and when it is clearly off the pitch (2 m past a line for 2 s).
- Touches (`touches.py`): a frame where the ball's velocity changes by more than 0.2 m/s with a player within 1 m. The ball track is smoothed, so at a sharp turn it cuts the corner and never reaches the foot; there we extend the straight paths in and out of the turn and accept a player within 1.5 m of where they meet (a weak touch). The touch goes to the nearest player's team, with fixes for the two keepers, whose tags the tracker sometimes swaps. A pass is two touches by different players of one team, at most 4 s apart, with the ball moving at least 2 m.
- Goals (`goal_celebration.py`): three or more players of one team running into the same point, followed within 4 minutes by both teams lining up for kick-off. The goal is dated 1 s before the run began.
- Kick-offs (`kick_off.py`): both teams on their own half and spread out. The taker is the team with the player nearest the centre spot, since only the kicking team may be in the centre circle.
- Bench breaks (`bench_break.py`): most players near the benches for at least 10 s, or fewer than 11 players tracked. The break runs from the walk to the benches to the walk back.
- Corner setups (`corner_setup.py`): at least 10 players walking inside one penalty box for at least 5 s.
- Ball in a box (`keeper_hold.py`): the ball tracked inside a penalty box, and each time the track loses the ball in a box and finds it again at that box.
- Smart Tagging (`smart_tagging.py`): the open-play phases say who has the ball (the named team for BUILD UP, PROGRESSION, CREATING CHANCES and ATTACKING TRANSITION, the other team for the DEFENDING phases). Set pieces give the restart type, the taker and the stoppage before it. The context codes (BALL IN THE BOX, LONG BALL and so on) are not used: their team does not say who has the ball.

### 4.2 Dead ball (D)

Combined in `resolve.py`, in this order:

- Start from the dead play found in the tracking.
- A keeper holding the ball in his box after a save is not a stoppage, even though everyone slows down.
- After a goal: dead from the goal until the kick-off is taken.
- Bench breaks are dead from the walk in to the walk out.
- A tagged set piece whose stoppage the tracking missed completely is taken from the tags.
- Before a kick-off: dead from the last long stoppage (a minute or more) to the kick-off.
- Before a corner: dead from the ball crossing the goal line to the corner kick, when a corner setup confirms it.
- A stoppage starts when the ball left the pitch, if that was up to 10 s before the players slowed down.
- A stoppage ends at the tagged restart when the ball is in play right after it, or at the kick or throw seen in the ball track.
- A ball clearly off the pitch is always dead. D wins over H and A.

### 4.3 Who has the ball (H / A)

- Possession stays with the last team until the other team has three touches in a row, each within 4 s of the previous one, or two touches when they complete a pass between two of their players. The change is then dated back to the first of those touches.
- Loose balls and challenges: one touch (a tackle, a block, a deflection, a header won) does not change possession. Touches only found at the corner of a turn are counted as weak and do not confirm a change on their own.
- A team's spell shorter than 3 s that runs straight into a stoppage is a clearance or a deflection, not control, so it goes back to the team that had the ball.
- After a stoppage, the team that restarts has the ball from the restart, not from its first detected touch.
- Kick-off: the team with the player nearest the centre spot is the taker and keeps the ball until the other team confirms possession as above, or until the next stoppage.
- Keeper: when the track loses the ball in a box, finds it again there, and the keeper's team plays it first with no stoppage after, the keeper's team has it from the moment the ball was lost.
- Missed shots: a fast kick that goes out over the goal line, within the width of the box, gives the shooting team the spell from their first touch to the ball going out, even with only one or two touches.
- Where there are no touches to go on, the Smart Tagging phases fill in.
- Frames still open are given the last H or A label in the same half. D is never carried forward. Frames outside the graded halves are X. On graded frames we always give an answer.

## 5. Limitations

The data:

- All the data we were given is automatic, not corrected, and most of our errors start there.
- The ball is missing on about 41% of graded frames. With no ball there are no touches, so changes of possession in these gaps are found late or missed.
- The player tracks have gaps and identity switches, sometimes between teams, so a touch can go to the wrong team.
- The Smart Tagging is inferred too. Its phases are not possessions, and its timings are only approximate.

Our method:

- A change needs two or three detected touches, so very short possessions are missed, and a change is only as early as the first touch we detect.
- Stoppages the ball track does not show, like a foul, are found from the players slowing down, which comes a few seconds after the whistle.
- The thresholds come from physical reasoning and were only checked against the Metrica sample games.


## 6. Reproduction

Python 3.10 or newer. Install the requirements:

```
pip install -r requirements.txt
```

Put the data bundle in `data/tactical/`, then run:

```
python scripts/make_submission.py
```

This writes `outputs/TikiData.csv`.
