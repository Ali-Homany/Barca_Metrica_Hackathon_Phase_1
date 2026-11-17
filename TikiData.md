# TikiData - More than a Hack 2027, Phase 1

Technical document. Ball possession for every frame of BAR-ATH.

## 1. Approach

Possession in football changes at identifiable events, such as an intercepted pass, a missed shot or a foul. We therefore treat the task as event detection at the possession level, not frame-by-frame classification. (focusing on optimizing for spellf1 not macrof1). The pipeline detects these events and features in the automatic tracking data (ATD) and the Smart Tagging, resolves them into a sequence of possessions and stoppages, and only then expands that sequence into the per-frame label file.

The difficulty of this problem lies in having good football knowledge, and turning this understanding into engineered features with thresholds, which is more direct than asking a model to relearn it from noisy tracking, and in our testing those rules were sufficient and achieved strong results.

Each rule is physically meaningful, can be inspected on its own, and can be switched off to measure its contribution.

The pipeline has two stages:
- Detection (src/detectors/): Independent detectors read the tracking and tagging data and report events such as ball touches, dead-ball intervals and restarts. Each works in isolation and does not see the others' output.
- Resolution (src/resolve.py): The place where detector outputs are combined and cross-detector rules are applied.

Then finally labels are carried forward to fill any remaining gaps.


## 2. Assets Used

- Tactical ATD (txt and xml).
- Tactical Smart Tagging (xml)
- Pitch size 105.3 x 68 m, from the given `games.py`.


## 3. Training data

We used no labelled dataset to train any model. The thresholds are set from physical reasoning (player and ball speeds, contact distances, penalty box size from the Laws of the Game), and some of them were tuned on the Metrica Sports open sample data, games 1 to 3. (https://github.com/metrica-sports/sample-data)

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

- A change needs two or three detected touches, so sometimes very short possessions are missed, and a change is only as early as the first touch we detect.
- Stoppages the ball track does not show, like a foul, are found from the players slowing down, which in some cases comes a few seconds late.
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
