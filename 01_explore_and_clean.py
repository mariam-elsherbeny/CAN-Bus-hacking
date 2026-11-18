# %% [markdown]
# # 01 - Explore & clean the HCRL Car-Hacking dataset
# Run cell by cell in VS Code (Run Cell above each `# %%`).
# Needs: pip install pandas numpy matplotlib pyarrow
#
# Expected layout:
#   data/raw/        <- the dataset files (DoS, Fuzzy, gear, RPM, normal)
#   data/processed/  <- cleaned parquet files (created here)
#   reports/figures/ <- plots (created here)

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path("data/raw")
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
# ## 1. Parsing
# Quirk of the attack CSVs: they have no header, and rows with DLC < 8 have
# fewer byte columns, so the flag (R/T) lands in the wrong column.
# Row format: timestamp, can_id, dlc, byte0..byte(dlc-1), flag
# We fix this by taking the LAST non-empty value of each row as the flag.

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


def load_csv(path: Path, attack: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, header=None, names=range(12), dtype=str, chunksize=500_000):
        parts.append(parse_chunk(chunk, attack))
    return pd.concat(parts, ignore_index=True)


LINE_RE = re.compile(r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*(\d)\s*(.*)")


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
# ## 2. Cleaning
# - sort by timestamp
# - drop exact duplicate rows (same timestamp + id + data)
# - sanity checks, then save to parquet

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
# ## 3. Sanity checks (what to look at before trusting the data)

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
# The paper says ~300 injections of 3-5 s each per file. Check that we see that.

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

# %% [markdown]
# ## 6. Things to write down in your notes / README
# - Which IDs dominate each attack (DoS: 0000 flood; Fuzzy: many random IDs; Gear/RPM: one real ID repeated).
# - Gear/RPM spoofing is the hardest: the injected messages look like real ones, so *timing* features matter.
# - Class imbalance per file (see plot 5a) -> use precision/recall/F1, not accuracy.
# - Next step: feature engineering (inter-arrival per ID, rolling window counts) and a time-based train/test split.
