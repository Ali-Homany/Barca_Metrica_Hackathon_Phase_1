"""Paths and match constants. Nothing else in the package should hard-code these."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# match bundle: set MTAH_DATA_DIR to point elsewhere, defaults to ./data
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data")
OUTPUT_DIR = ROOT / "outputs"

TEAM_NAME = "TikiData"  # the csv is saved as <TEAM_NAME>.csv

FPS = 25
N_FRAMES = 199_319  # tactical video, one submission row per frame

# (first_frame, last_frame) inclusive, only these are graded
HALVES = {1: (11_185, 90_651), 2: (110_808, 190_197)}

PITCH_LENGTH = 105.3  # metres, measured from the corrected tracking
PITCH_WIDTH = 68.0

LABELS = ("H", "A", "D", "X")

# club name as written in the data, mapped to its side in the submission
TEAM_SIDES = {"Barcelona": "H", "Atletic Club": "A"}


def tactical_paths() -> dict[str, Path]:
    base = DATA_DIR / "tactical"
    return {
        "video": base / "BAR-ATH_tactical.mp4",
        "atd_meta": base / "BAR-ATH_tactical_atd.xml",
        "atd_raw": base / "BAR-ATH_tactical_atd.txt",
        "homography": base / "BAR-ATH_tactical_homography.csv",
        "smart_tagging": base / "BAR-ATH_tactical_smart_tagging.xml",
    }
