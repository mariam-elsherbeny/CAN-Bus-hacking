# %% [markdown]
# ## 0. Environment Setup & Configuration
#
# This section prepares the environment and defines the paths,
# dataset filenames, and constants used throughout the notebook.
#
# The HCRL Car-Hacking dataset contains four attack scenarios
# (DoS, Fuzzy, Gear, RPM) and normal driving data.
#
# All datasets will be loaded, processed, and exported for
# further exploration and model development.

# %%
import sys
print(sys.executable)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
import pyarrow
print(pyarrow.__version__)

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path("data/car-hacking-dataset/Car-Hacking")
OUT = Path("data/processed")
FIG = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# name -> file name. EDIT these if your files are named differently.
FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
    "Normal": "normal_run_data.txt",
}

BYTE_COLS = [f"b{i}" for i in range(8)]
HEX = {f"{i:02x}": i for i in range(256)} | {f"{i:02X}": i for i in range(256)}

print("Files found in data/raw:", [p.name for p in RAW.iterdir()] if RAW.exists() else "FOLDER MISSING")

# %% [markdown]
# %% [markdown]
# ## 1. Data Parsing
#
# The original dataset files have different formats:
# - Attack datasets are stored as CSV files without headers.
# - Normal driving data is stored in a TXT file.
#
# This section converts both formats into a consistent
# Pandas DataFrame structure.
#
# Each CAN message contains:
# - Timestamp
# - CAN ID
# - DLC (Data Length Code)
# - Up to 8 data bytes
# - Label (Normal / Attack)
# - Attack type
#
# Special handling is required for attack CSVs because
# messages with DLC < 8 shift the flag position.
# %%
def parse_chunk(chunk: pd.DataFrame, attack: str) -> pd.DataFrame:
    flag = chunk.iloc[:, 3:].ffill(axis=1).iloc[:, -1]
    data = chunk.iloc[:, 3:11].replace({"R": np.nan, "T": np.nan})

    ids = chunk[1].astype(str).str.strip()
    id_map = {u: int(u, 16) for u in ids.unique()}

    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(chunk[0]),
            "can_id": ids.map(id_map).astype("int16"),
            "dlc": pd.to_numeric(chunk[2]).astype("int8"),
        }
    )
    for i, c in enumerate(BYTE_COLS):
        out[c] = data.iloc[:, i].map(HEX).fillna(0).astype("uint8")
    out["label"] = (flag.str.strip() == "T").astype("int8")
    out["attack_type"] = attack
    return out
# %% [markdown]
# ### 1.1 Parsing Attack CSV Chunks
#
# Processes the attack CSV files in chunks to reduce memory usage.
#
# The function extracts the transmission flag, converts hexadecimal
# CAN IDs and data bytes into numeric values, and assigns labels
# according to the message flag.

def load_csv(path: Path, attack: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, header=None, names=range(12), dtype=str, chunksize=500_000):
        parts.append(parse_chunk(chunk, attack))
    return pd.concat(parts, ignore_index=True)


LINE_RE = re.compile(r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*(\d)\s*(.*)")

# %% [markdown]
# ### 1.3 Parsing Normal Driving Data
# The normal dataset uses a different text-based format.
# Regular expressions are used to extract CAN message fields.
# The extracted messages are converted into the same structure
# used for the attack datasets and assigned a normal label.
def load_normal_txt(path: Path) -> pd.DataFrame:
    """normal_run_data.txt uses a different text format than the CSVs."""
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ts, cid, dlc, rest = m.groups()
            b = [HEX.get(x, 0) for x in rest.split()][:8]
            b += [0] * (8 - len(b))
            rows.append((float(ts), int(cid, 16), int(dlc), *b))
    df = pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", *BYTE_COLS])
    df = df.astype({"can_id": "int16", "dlc": "int8", **{c: "uint8" for c in BYTE_COLS}})
    df["label"] = np.int8(0)
    df["attack_type"] = "Normal"
    return df


def load(name: str) -> pd.DataFrame:
    path = RAW / FILES[name]
    if path.suffix == ".txt":
        return load_normal_txt(path)
    return load_csv(path, name)


# %% [markdown]
# ## 2. Data Cleaning & Preprocessing
#
# After parsing, the datasets undergo basic cleaning to ensure
# consistency and improve data quality.
#
# The cleaning process includes:
# - Sorting CAN messages chronologically.
# - Removing duplicate messages based on timestamp, CAN ID,
#   DLC, and data bytes.
# - Converting attack_type into a categorical data type.
#
# The cleaned datasets are then exported in Parquet format
# for efficient storage and future processing.

# %%
def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp", "can_id", "dlc", *BYTE_COLS], keep="first").reset_index(drop=True)
    df["attack_type"] = df["attack_type"].astype("category")
    print(f"   dropped {n0 - len(df):,} duplicate rows")
    return df


data = {}
for name, fname in FILES.items():
    if not (RAW / fname).exists():
        print(f"[skip] {fname} not found")
        continue
    print(f"Loading {name} ...")
    df = clean(load(name))
    df.to_parquet(OUT / f"{name.lower()}.parquet", index=False)
    data[name] = df
    print(f"   {len(df):,} rows saved -> {OUT / (name.lower() + '.parquet')}")

# %% [markdown]
# %% [markdown]
# ## 3. Data Validation & Sanity Checks
#
# Before proceeding with exploratory analysis, we inspect
# the cleaned datasets to verify their basic integrity.
#
# The checks include:
# - Total number of messages.
# - Recording duration.
# - Number of unique CAN IDs.
# - Missing values.
# - Timestamp ordering.
# - Normal vs. attack label distribution.
# - Distribution of DLC values.
#
# These checks help identify potential data quality issues
# before moving to deeper analysis.
# %%
for name, df in data.items():
    dur = df.timestamp.iloc[-1] - df.timestamp.iloc[0]
    print(f"\n=== {name} ===")
    print(f"rows: {len(df):,} | duration: {dur / 60:.1f} min | unique IDs: {df.can_id.nunique()}")
    print(f"nulls: {int(df.isna().sum().sum())} | timestamps sorted: {df.timestamp.is_monotonic_increasing}")
    print("labels:", df.label.value_counts().to_dict(), f"| attack share: {df.label.mean():.2%}")
    print("DLC counts:", df.dlc.value_counts().sort_index().to_dict())

# %% [markdown]
# ## 4. Attack segments
# %% [markdown]
# ## 4. Attack Segment Analysis
#
# Attack messages are not necessarily distributed continuously
# throughout a recording. Instead, they may appear in distinct
# periods of attack activity.
#
# This section identifies consecutive attack segments using
# changes in the message labels.
#
# For each segment, we examine:
# - Starting timestamp.
# - Ending timestamp.
# - Number of messages.
# - Attack duration.
#
# The results help us understand the temporal distribution
# of attack activity within each dataset.
# %%
def attack_segments(df: pd.DataFrame) -> pd.DataFrame:
    run_id = (df.label != df.label.shift()).cumsum()
    seg = df[df.label == 1].groupby(run_id[df.label == 1]).agg(
        start=("timestamp", "min"), end=("timestamp", "max"), n_msgs=("timestamp", "size")
    )
    seg["duration_s"] = seg.end - seg.start
    return seg


segments = {}
for name, df in data.items():
    if name == "Normal":
        continue
    seg = attack_segments(df)
    segments[name] = seg
    print(f"\n{name}: {len(seg)} segments | duration median {seg.duration_s.median():.1f}s "
          f"(min {seg.duration_s.min():.1f}, max {seg.duration_s.max():.1f}) | msgs/segment median {int(seg.n_msgs.median()):,}")
    # NOTE: short gaps inside a long attack can split it into extra segments - that's fine, just be aware.

# %% [markdown]
# ## 5. Plots

# %%
# 5a. Class balance
names = [n for n in data if n != "Normal"]
fig, ax = plt.subplots(figsize=(7, 4))
counts = pd.DataFrame({n: data[n].label.value_counts() for n in names}).T.rename(columns={0: "normal", 1: "attack"})
counts.plot.bar(stacked=True, ax=ax, color=["#4c9f70", "#d9534f"])
ax.set_title("Messages per file: normal vs attack")
ax.set_ylabel("messages")
plt.tight_layout()
plt.savefig(FIG / "class_balance.png", dpi=150)
plt.show()

# %%
# 5b. Messages per second over time, attacks shaded
for name in names:
    df = data[name]
    t = df.timestamp - df.timestamp.iloc[0]
    rate = df.groupby(t.astype(int)).size()
    att = df[df.label == 1]
    att_rate = att.groupby((att.timestamp - df.timestamp.iloc[0]).astype(int)).size().reindex(rate.index, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rate.index, rate.values, lw=0.8, label="all messages/s")
    ax.fill_between(att_rate.index, 0, att_rate.values, color="#d9534f", alpha=0.5, label="attack messages/s")
    ax.set_title(f"{name}: message rate over time")
    ax.set_xlabel("seconds since start")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"rate_{name.lower()}.png", dpi=150)
    plt.show()

# %%
# 5c. Top CAN IDs: normal vs attack
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=False)
for ax, name in zip(np.atleast_1d(axes), names):
    df = data[name]
    top = df[df.label == 1].can_id.value_counts().head(8)
    top.index = [f"{i:04x}" for i in top.index]
    top.plot.bar(ax=ax, color="#d9534f")
    ax.set_title(f"{name}: top attack IDs")
plt.tight_layout()
plt.savefig(FIG / "top_attack_ids.png", dpi=150)
plt.show()

# %%
# 5d. Inter-arrival time (time between consecutive messages), normal vs attack
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5))
for ax, name in zip(np.atleast_1d(axes), names):
    df = data[name]
    dt = df.timestamp.diff().clip(lower=1e-6)
    for lab, color, text in [(0, "#4c9f70", "normal"), (1, "#d9534f", "attack")]:
        vals = dt[df.label == lab].dropna()
        if len(vals):
            ax.hist(np.log10(vals), bins=80, alpha=0.6, color=color, label=text, density=True)
    ax.set_title(f"{name}")
    ax.set_xlabel("log10(seconds between messages)")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "interarrival.png", dpi=150)
plt.show()


# %%
# %% [markdown]
# ## 6. Recording Boundary Detection
#
# A dataset file may contain multiple recording periods
# separated by large gaps in timestamps.
#
# A time gap threshold is used to distinguish separate
# recording periods, and each message is assigned a recording ID.

for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    dt = data[name].timestamp.diff()
    print(name, "largest gap (s):", round(dt.max(), 1), "at row", dt.idxmax())
# %%
GAP = 10  # seconds; anything bigger = a new recording

# %% [markdown]
# ### 6.1 Comparing Recording Tails
#
# The final recording period of each attack dataset is extracted
# and compared against the corresponding DoS dataset tail.
#
# This checks whether the same normal driving sequence is
# present across the different attack files.

for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    df = data[name]
    df["recording"] = (df.timestamp.diff() > GAP).cumsum().astype("int8")
    print(f"\n{name}")
    print(df.groupby("recording").agg(
        rows=("label", "size"),
        attack_share=("label", "mean"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    ))
# %%
cols = ["timestamp", "can_id", "dlc", *BYTE_COLS]
tails = {n: data[n][data[n].recording == 1].reset_index(drop=True)[cols]
         for n in ["DoS", "Fuzzy", "Gear", "RPM"]}
for n, t in tails.items():
    print(n, "tail identical to DoS tail:", t.equals(tails["DoS"]))
# %% # %% [markdown]
# ### 6.2 Constructing Separate Normal and Attack Datasets
#
# Based on the recording analysis, the identified normal tail
# is extracted from the DoS dataset and assigned the Normal label.
# The resulting Parquet files provide a separated representation
# of normal and attack traffic for subsequent experiments.

normal_tail = data["DoS"][data["DoS"].recording == 1].drop(columns="recording").reset_index(drop=True)
normal_tail["attack_type"] = "Normal"
normal_tail.to_parquet(OUT / "normal_tail.parquet", index=False)

for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    rec0 = data[name][data[name].recording == 0].drop(columns="recording").reset_index(drop=True)
    rec0.to_parquet(OUT / f"{name.lower()}_attack.parquet", index=False)
    print(name, len(rec0), "rows")
print("normal tail:", len(normal_tail), "rows")
# 
# %% [markdown]
# ## 7. Attack Episode Analysis
#
# Attack segments identified through label transitions may
# contain multiple consecutive periods of malicious activity.
# This section applies a time-based approach to identify
# attack episodes.
#
# A new episode is considered to begin when the time difference
# between consecutive attack messages exceeds the specified
# maximum gap.
#
# For each episode, we calculate:
# - Starting timestamp.
# - Ending timestamp.
# - Number of attack messages.
# - Episode duration.
#
# This provides another perspective on the temporal structure
# and frequency of attack activity.

def attack_episodes(df, max_gap=1.0):
    t = df.loc[df.label == 1, "timestamp"]
    ep = (t.diff() > max_gap).cumsum()          # new episode if >1 s since last attack msg
    g = t.groupby(ep).agg(start="min", end="max", n_msgs="size")
    g["duration_s"] = g.end - g.start
    return g

for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    eps = attack_episodes(data[name][data[name].recording == 0])
    print(f"{name}: {len(eps)} episodes | median {eps.duration_s.median():.1f}s "
          f"(min {eps.duration_s.min():.1f}, max {eps.duration_s.max():.1f})")
# 