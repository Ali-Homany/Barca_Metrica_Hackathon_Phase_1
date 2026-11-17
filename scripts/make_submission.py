# %% [markdown]
# # Make the submission
# Runs the pipeline and writes `outputs/<team_name>.csv`.
# Works as a plain script (`python scripts/make_submission.py`) or cell by cell as a notebook (jupytext).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, pipeline

# %%
df = pipeline.run()

# %%
assert len(df) == config.N_FRAMES, f"expected {config.N_FRAMES} rows, got {len(df)}"
assert df["label"].isin(config.LABELS).all(), "labels must be one of H A D X"
assert (df["frame"].to_numpy() == range(config.N_FRAMES)).all(), "frames must be 0..N-1 in order"

# %%
config.OUTPUT_DIR.mkdir(exist_ok=True)
out = config.OUTPUT_DIR / f"{config.TEAM_NAME}.csv"
df.to_csv(out, index=False)
print(f"wrote {out}  ({df['label'].value_counts().to_dict()})")
