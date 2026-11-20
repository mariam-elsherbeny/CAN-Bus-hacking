# %% [markdown]
# # Stage 2 - Feature Engineering
#
# Input : data/processed/{dos,fuzzy,gear,rpm}_attack.parquet + normal_tail.parquet
# Output: data/features/<name>_features.parquet + baseline_id_stats.parquet
#
# Rules followed everywhere:
#  - every feature is CAUSAL: it only uses the current and PAST messages
#    (so the same code works later in the live streaming replay).
#  - history windows exclude the current message when they describe "what is normal".
#  - baseline stats (normal period per ID, normal DLC) come from the first
#    BASELINE_FRAC of the normal tail only. Do NOT use that part of the tail
#    as test data later.

# %%
from common import *   # np, pd, json, Path, PROC, FEAT, BYTE_COLS, META_COLS ...
W_SHORT, W_LONG = 0.1, 1.0      # time windows in seconds
N_WIN = 50                      # message-count window (global)
K_HIST = 10                     # per-ID history length (messages)
BASELINE_FRAC = 0.6
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# %% [markdown]
# ## 1. Helpers

# %%
def window_counts(ts: np.ndarray, w: float) -> np.ndarray:
    """Messages in (t - w, t], current included. ts must be sorted."""
    left = np.searchsorted(ts, ts - w, side="left")
    return (np.arange(len(ts)) - left + 1).astype(np.int32)


def trailing_mean_std(x: np.ndarray, k: int):
    """Mean/std over the PREVIOUS k rows (current row excluded). x: (m, c)."""
    x = x.astype(np.float64)
    zero = np.zeros((1, x.shape[1]))
    cs = np.vstack([zero, np.cumsum(x, axis=0)])
    cs2 = np.vstack([zero, np.cumsum(x ** 2, axis=0)])
    i = np.arange(len(x))
    lo = np.maximum(0, i - k)
    cnt = (i - lo).astype(np.float64)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cs[i] - cs[lo]) / cnt
        var = (cs2[i] - cs2[lo]) / cnt - mean ** 2
    return mean, np.sqrt(np.clip(var, 0, None))   # row 0 -> NaN (no history)


# %% [markdown]
# ## 2. Baseline from normal traffic
# Normal period (median inter-arrival) and DLC for every ID.
# Fuzzy attacks inject random IDs -> unknown IDs; DoS floods ID 0x000 at a
# much shorter period than normal.

# %%
def build_baseline(normal: pd.DataFrame, frac: float = BASELINE_FRAC):
    normal = normal.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut = float(normal.timestamp.iloc[int(len(normal) * frac)])
    base = normal[normal.timestamp < cut]
    dt = base.groupby("can_id").timestamp.diff()
    span = base.timestamp.max() - base.timestamp.min()
    stats = pd.DataFrame({
        "base_period": dt.groupby(base.can_id).median(),
        "base_dlc": base.groupby("can_id").dlc.agg(lambda s: s.mode().iloc[0]),
        "base_rate": base.groupby("can_id").size() / span,
    })
    return stats, cut


# %% [markdown]
# ## 3. Feature builder
# Groups of features:
#  A. inter-arrival time (global + per ID)
#  B. per-ID frequency
#  C. rolling-window statistics (time- and count-based)
#  D. payload differences vs the previous message of the same ID
#  E. temporal / signal-level (byte values vs the ID's recent history)
#  F. baseline-aware (known ID, DLC mismatch, period ratio)

# %%
def build_features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    ts = df.timestamp.to_numpy()
    cid = df.can_id.to_numpy()
    dlc = df.dlc.to_numpy()
    B = df[BYTE_COLS].to_numpy(np.uint8)
    pos = np.arange(n)

    # ---- A/C. global inter-arrival + global rolling ------------------------
    dt_global = np.diff(ts, prepend=np.nan)
    s = pd.Series(dt_global)
    roll = s.rolling(N_WIN)
    F = {
        "dt_global": dt_global,
        f"dt_global_mean_{N_WIN}": roll.mean().to_numpy(),
        f"dt_global_std_{N_WIN}": roll.std().to_numpy(),
        "rate_100ms": window_counts(ts, W_SHORT),
        "rate_1s": window_counts(ts, W_LONG),
    }

    # ---- per-ID pass --------------------------------------------------------
    dt_id = np.full(n, np.nan)
    dt_hist_mean = np.full(n, np.nan)
    dt_hist_std = np.full(n, np.nan)
    id_cnt_100ms = np.zeros(n, np.int32)
    id_cnt_1s = np.zeros(n, np.int32)
    id_in_last_n = np.zeros(n, np.int32)
    byte_hist = np.full((n, 8), np.nan)
    prev_B = np.zeros_like(B)
    has_prev = np.zeros(n, bool)
    nxt = np.full(n, n)                       # index of next msg with same ID

    for _, idx in df.groupby("can_id").indices.items():
        t = ts[idx]
        id_cnt_100ms[idx] = window_counts(t, W_SHORT)
        id_cnt_1s[idx] = window_counts(t, W_LONG)
        # how many of the last N_WIN global messages had this ID (incl. current)
        id_in_last_n[idx] = np.arange(len(idx)) - np.searchsorted(idx, idx - N_WIN + 1, side="left") + 1
        if len(idx) < 2:
            continue
        d = np.diff(t)
        dt_id[idx[1:]] = d
        nxt[idx[:-1]] = idx[1:]
        prev_B[idx[1:]] = B[idx[:-1]]
        has_prev[idx[1:]] = True
        m, sd = trailing_mean_std(d[:, None], K_HIST)
        dt_hist_mean[idx[1:]] = m[:, 0]
        dt_hist_std[idx[1:]] = sd[:, 0]
        byte_hist[idx], _ = trailing_mean_std(B[idx], K_HIST)

    # unique IDs among the last N_WIN messages (difference-array trick)
    ends = np.bincount(np.minimum(nxt, pos + N_WIN), minlength=n + 1)
    uniq_ids = np.cumsum(1 - ends[:n]).astype(np.int32)

    # ---- A/B. per-ID inter-arrival and frequency ---------------------------
    F.update({
        "dt_id": dt_id,
        "dt_id_hist_mean": dt_hist_mean,
        "dt_id_hist_std": dt_hist_std,
        "dt_id_vs_hist": dt_id / dt_hist_mean,
        "id_cnt_100ms": id_cnt_100ms,
        "id_cnt_1s": id_cnt_1s,
        f"id_share_last_{N_WIN}": id_in_last_n / np.minimum(pos + 1, N_WIN),
        f"unique_ids_last_{N_WIN}": uniq_ids,
        "id_first_seen": (~has_prev).astype(np.int8),
    })

    # ---- D. payload differences vs previous same-ID message ----------------
    xor = B ^ prev_B
    bit_flips = POPCOUNT[xor].sum(axis=1).astype(np.float32)
    n_changed = (xor != 0).sum(axis=1).astype(np.float32)
    abs_diff = np.abs(B.astype(np.int16) - prev_B.astype(np.int16))
    for name, arr in [("bit_flips", bit_flips), ("bytes_changed", n_changed),
                      ("abs_diff_sum", abs_diff.sum(axis=1).astype(np.float32)),
                      ("abs_diff_max", abs_diff.max(axis=1).astype(np.float32))]:
        arr = arr.copy()
        arr[~has_prev] = np.nan
        F[name] = arr
    for i, c in enumerate(BYTE_COLS):
        a = abs_diff[:, i].astype(np.float32)
        a[~has_prev] = np.nan
        F[f"{c}_absdiff"] = a

    # ---- E. signal-level: payload shape + deviation from ID history --------
    F["byte_mean"] = B.mean(axis=1)
    F["byte_std"] = B.std(axis=1)
    F["zero_bytes"] = (B == 0).sum(axis=1).astype(np.int8)
    dev = np.abs(B - byte_hist)
    F["hist_dev_sum"] = dev.sum(axis=1)
    F["hist_dev_max"] = dev.max(axis=1)

    # ---- F. baseline-aware ---------------------------------------------------
    cs = pd.Series(cid)
    base_period = cs.map(base.base_period).to_numpy(dtype=float)
    base_dlc = cs.map(base.base_dlc).to_numpy(dtype=float)
    known = cs.isin(base.index).to_numpy()
    F["is_known_id"] = known.astype(np.int8)
    F["dlc_mismatch"] = np.where(known, dlc != base_dlc, 1).astype(np.int8)
    F["dt_id_vs_base"] = dt_id / base_period

    feats = pd.DataFrame(F).astype("float32")
    meta = df[["timestamp", "can_id", "dlc", *BYTE_COLS, "label", "attack_type"]]
    return pd.concat([meta, feats], axis=1)


# %% [markdown]
# ## 4. Run on every recording

# %%
normal = pd.read_parquet(PROC / "normal_tail.parquet")
base, cut = build_baseline(normal)
base.to_parquet(FEAT / "baseline_id_stats.parquet")
json.dump({"baseline_cut_timestamp": cut, "baseline_frac": BASELINE_FRAC},
          open(FEAT / "baseline_meta.json", "w"))
print(f"baseline: {len(base)} IDs, cut at t={cut:.1f}")

sources = {
    "normal": "normal_tail.parquet",
    "dos": "dos_attack.parquet",
    "fuzzy": "fuzzy_attack.parquet",
    "gear": "gear_attack.parquet",
    "rpm": "rpm_attack.parquet",
}
feat = {}
for name, fname in sources.items():
    print(f"building {name} ...")
    df = normal if name == "normal" else pd.read_parquet(PROC / fname)
    feat[name] = build_features(df, base)
    feat[name].to_parquet(FEAT / f"{name}_features.parquet", index=False)
    print(f"   {feat[name].shape[0]:,} rows x {feat[name].shape[1]} cols")

FEATURE_COLS = [c for c in feat["dos"].columns
                if c not in META_COLS]
json.dump(FEATURE_COLS, open(FEAT / "feature_cols.json", "w"), indent=2)
print(len(FEATURE_COLS), "engineered features")

# %% [markdown]
# ## 5. Sanity check: do features separate attack from normal?
# Mean of key features by label, per attack type.

# %%
KEY = ["dt_global", "dt_id", "dt_id_vs_base", "id_cnt_1s", "rate_1s",
       "unique_ids_last_50", "bit_flips", "hist_dev_sum", "is_known_id", "dlc_mismatch"]
for name in ["dos", "fuzzy", "gear", "rpm"]:
    print(f"\n=== {name} ===")
    print(feat[name].groupby("label")[KEY].mean().T.round(4))
