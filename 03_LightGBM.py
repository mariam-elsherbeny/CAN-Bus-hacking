# %% [markdown]
# # Stage 3 - Time-block split + LightGBM baseline (one model per attack)
#
# Input : data/features/<name>_features.parquet, normal_features.parquet,
#         baseline_meta.json, feature_cols.json   (from Stage 2)
# Output: models/lgbm_<name>.txt, reports/metrics.csv,
#         reports/episodes_<name>.csv, reports/figures/importance_<name>.png
#
# Split rules:
#  - drop the first WARMUP rows of each recording (rolling features not warmed up)
#  - drop the part of the normal tail used for the baseline (first 60%)
#  - contiguous time blocks: 70% train / 15% val / 15% test, inside each recording
#  - raw can_id and raw bytes are NOT model inputs, only engineered features
#
# Reported twice:
#  - "attack recording": the test block of the attack recording only (the honest,
#    harder number: attacks and regular messages interleaved)
#  - "with normal tail": adds the clean normal test block (easy negatives, so
#    precision looks better)
# %%
import warnings

import lightgbm as lgb
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)

from common import *

warnings.filterwarnings("ignore", module="lightgbm")   # eval_set deprecation notice in newer versions

MODELS = Path("models")
REPORTS = Path("reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

WARMUP = 50
SPLIT = (0.70, 0.15, 0.15)
EPISODE_GAP = 1.0        # s; same rule as Stage 1 attack episodes

FEATURE_COLS = json.load(open(FEAT / "feature_cols.json"))
CUT = json.load(open(FEAT / "baseline_meta.json"))["baseline_cut_timestamp"]


# %% [markdown]
# ## 1. Split

# %%
def block_split(df: pd.DataFrame, fr=SPLIT):
    n = len(df)
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def make_splits(name: str, normal_free: pd.DataFrame):
    att = pd.read_parquet(FEAT / f"{name}_features.parquet")
    att = att.sort_values("timestamp", kind="stable").iloc[WARMUP:].reset_index(drop=True)
    a_tr, a_va, a_te = block_split(att)
    n_tr, n_va, n_te = block_split(normal_free)
    return {
        "train": pd.concat([a_tr, n_tr], ignore_index=True),
        "val": pd.concat([a_va, n_va], ignore_index=True),
        "test_att": a_te.reset_index(drop=True),
        "test_norm": n_te.reset_index(drop=True),
    }


# %% [markdown]
# ## 2. Metrics
# Message level: precision / recall / F1 / PR-AUC.
# Episode level: was each attack episode caught, and how fast?
#   latency = time from the first injected message of an episode to the first
#   alert raised inside that episode.

# %%
def best_threshold(y, proba):
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def msg_metrics(y, pred, proba):
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba) if y.sum() else np.nan,
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def episode_report(ts, y, pred, gap=EPISODE_GAP) -> pd.DataFrame:
    att = np.flatnonzero(y == 1)
    if len(att) == 0:
        return pd.DataFrame(columns=["start", "end", "detected", "latency_s"])
    t_att = ts[att]
    ep = np.concatenate([[0], np.cumsum(np.diff(t_att) > gap)])
    rows = []
    for e in range(ep.max() + 1):
        te = t_att[ep == e]
        s, en = te[0], te[-1]
        lo, hi = np.searchsorted(ts, s, "left"), np.searchsorted(ts, en, "right")
        alerts = np.flatnonzero(pred[lo:hi] == 1)
        rows.append((s, en, len(alerts) > 0, ts[lo + alerts[0]] - s if len(alerts) else np.nan))
    return pd.DataFrame(rows, columns=["start", "end", "detected", "latency_s"])


# %% [markdown]
# ## 3. Train + evaluate every attack

# %%
normal = pd.read_parquet(FEAT / "normal_features.parquet")
normal_free = (normal[normal.timestamp >= CUT]
               .sort_values("timestamp", kind="stable").reset_index(drop=True))
print(f"normal tail rows usable for train/val/test: {len(normal_free):,} "
      f"(baseline part excluded)")

results, importances = [], {}
for atk in ATTACKS:
    name = atk.lower()
    print(f"\n=== {atk} ===")
    sp = make_splits(name, normal_free)
    Xtr, ytr = sp["train"][FEATURE_COLS], sp["train"].label.to_numpy()
    Xva, yva = sp["val"][FEATURE_COLS], sp["val"].label.to_numpy()
    print(f"train {len(Xtr):,} (attack {ytr.mean():.1%}) | val {len(Xva):,} | "
          f"test attack-recording {len(sp['test_att']):,} + normal {len(sp['test_norm']):,}")

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    print(f"best iteration: {model.best_iteration_}")
    model.booster_.save_model(str(MODELS / f"lgbm_{name}.txt"))

    thr = best_threshold(yva, model.predict_proba(Xva)[:, 1])   # chosen on validation only

    for label, df in [("attack recording", sp["test_att"]),
                      ("with normal tail", pd.concat([sp["test_att"], sp["test_norm"]],
                                                     ignore_index=True))]:
        y = df.label.to_numpy()
        proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
        pred = (proba >= thr).astype(int)
        m = msg_metrics(y, pred, proba)
        row = {"attack": atk, "test_set": label, "threshold": round(thr, 4), **m}

        if label == "attack recording":
            eps = episode_report(df.timestamp.to_numpy(), y, pred)
            eps.to_csv(REPORTS / f"episodes_{name}.csv", index=False)
            lat = eps.latency_s.dropna() * 1000
            dur_min = (df.timestamp.iloc[-1] - df.timestamp.iloc[0]) / 60
            row.update({
                "episodes": len(eps),
                "episodes_detected": float(eps.detected.mean()) if len(eps) else np.nan,
                "latency_median_ms": lat.median() if len(lat) else np.nan,
                "latency_p95_ms": lat.quantile(0.95) if len(lat) else np.nan,
                "false_alarms_per_min": m["fp"] / dur_min,
            })
        results.append(row)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURE_COLS)
    importances[atk] = (imp / imp.sum()).sort_values(ascending=False)
    top = importances[atk].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    top.plot.barh(ax=ax, color="#4c9f70")
    ax.set_title(f"{atk}: top features (share of gain)")
    plt.tight_layout()
    plt.savefig(FIG / f"importance_{name}.png", dpi=150)
    plt.close(fig)

# %% [markdown]
# ## 4. Summary

# %%
res = pd.DataFrame(results)
res.to_csv(REPORTS / "metrics.csv", index=False)
pd.set_option("display.width", 200)
print("\n--- message-level metrics ---")
print(res[["attack", "test_set", "precision", "recall", "f1", "pr_auc", "tp", "fp", "fn"]]
      .round(4).to_string(index=False))
print("\n--- episode-level (attack recording test block) ---")
print(res[res.test_set == "attack recording"]
      [["attack", "episodes", "episodes_detected", "latency_median_ms",
        "latency_p95_ms", "false_alarms_per_min"]].round(3).to_string(index=False))
print("\n--- top 5 features per attack ---")
for atk, imp in importances.items():
    print(atk, "->", ", ".join(f"{k} ({v:.0%})" for k, v in imp.head(5).items()))

# %%
