"""Load the Smart Tagging XML (SportsCode export) as a table of video-frame intervals.

Time base, verified by the organisers' starter/load_smart_tagging.py: <start>/<end> are seconds from the
first frame of the tactical video, so seconds x FPS is a video frame directly.
"""
import xml.etree.ElementTree as ET
from functools import cache

import pandas as pd

from .. import config

NO_TEAM = "N/A"  # Team value on phases without a team (every throw-in, kick-offs, a few others)

# Every tag group in the file, mapped to the column it is kept as, or None when it says nothing about who has
# the ball or whether it is dead. An unknown group raises, so a changed export is noticed.
GROUPS = {
    "Team": "team",  # club name, mapped to 'H' / 'A' (missing when N/A)
    "Type": "type",  # which restart, set pieces only
    "Half": "half",
    "Side": None,  # Central / Left / Right: where the play happens
    "Direction of ball entry": None,  # Horizontal / Vertical / Diagonal: how the ball entered the box
    "Max Players in the box": None,  # 1..7+: how many attackers were in the box
}


@cache
def load() -> pd.DataFrame:
    """One row per tagged instance, sorted by start frame (file order breaks ties).

    Columns: code (upper case), team ('H' / 'A' / missing), type (set pieces only, else missing), half,
    start_frame, end_frame. Intervals are half-open: end_frame is the first frame after the phase.
    Cached: treat the result as read-only (use .copy() before modifying it).
    """
    rows = []
    for inst in ET.parse(config.tactical_paths()["smart_tagging"]).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        if unknown := tags.keys() - GROUPS.keys():
            raise ValueError(f"unknown tag group(s) {unknown}: decide how to treat them in GROUPS")
        team = tags.get("Team")
        if team not in config.TEAM_SIDES and team not in (NO_TEAM, None):
            raise ValueError(f"unknown Team tag {team!r}: add it to config.TEAM_SIDES")
        row = {col: tags.get(group) for group, col in GROUPS.items() if col}
        row.update({
            "code": inst.findtext("code").strip().upper(),
            "team": config.TEAM_SIDES.get(team),
            "start_frame": round(float(inst.findtext("start")) * config.FPS),
            "end_frame": round(float(inst.findtext("end")) * config.FPS),
        })
        rows.append(row)
    cols = ["code", "team", "type", "half", "start_frame", "end_frame"]
    return pd.DataFrame(rows)[cols].sort_values("start_frame", kind="stable", ignore_index=True)
