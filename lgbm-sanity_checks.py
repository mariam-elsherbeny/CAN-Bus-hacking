# %% [markdown]
# # Stage 3b - Are the near-perfect scores real?
#
# Three checks on the Stage 3 results:
#  1. Feature-group ablation: retrain without / with only each feature group.
#     Shows which signal the model really needs.
#  2. First-injected-message check: is the FIRST message of an episode already
#     flagged (latency 0)? What makes it different from regular messages?
#  3. Cross-attack transfer: train on one attack, test on another. Shows whether the
#     model learned a general injection signature or memorised one attack.
#
# Uses a 300k-row training sample and 200 trees per fit so everything runs in
# minutes. Threshold 0.5 (no tuning). Needs models/ and reports/metrics.csv
# from 03_train_baseline.py.

# %%
import warnings

import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score

from common import *

warnings.filterwarnings("ignore", module="lightgbm")

WARMUP, SPLIT, TRAIN_SAMPLE, EPISODE_GAP = 50, (0.70, 0.15, 0.15), 300_000, 1.0
FEATURE_COLS = json.load(open(FEAT / "feature_cols.json"))
CUT = json.load(open(FEAT / "baseline_meta.json"))["baseline_cut_timestamp"]

GROUPS = {
    "timing_global": ["dt_global", "dt_global_mean_50", "dt_global_std_50", "rate_100ms", "rate_1s"],
    "per_id_timing": ["dt_id", "dt_id_hist_mean", "dt_id_hist_std", "dt_id_vs_hist", "dt_id_vs_base"],
    "per_id_frequency": ["id_cnt_100ms", "id_cnt_1s", "id_share_last_50", "unique_ids_last_50", "id_first_seen"],
    "payload_change": ["bit_flips", "bytes_changed", "abs_diff_sum", "abs_diff_max"]
                      + [f"b{i}_absdiff" for i in range(8)],
    "payload_shape": ["byte_mean", "byte_std", "zero_bytes", "hist_dev_sum", "hist_dev_max"],
    "identity": ["is_known_id", "dlc_mismatch"],
}
GROUPS = {g: [c for c in cols if c in FEATURE_COLS] for g, cols in GROUPS.items()}
missing = set(FEATURE_COLS) - {c for cols in GROUPS.values() for c in cols}
if missing:
    print("WARNING: features not in any group:", sorted(missing))


# %% [markdown]
# ## Data and helpers

# %%
def block_split(df, fr=SPLIT):
    n = len(df)
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


normal = pd.read_parquet(FEAT / "normal_features.parquet")
normal_free = (normal[normal.timestamp >= CUT].sort_values("timestamp", kind="stable")
               .reset_index(drop=True))
n_train, _, _ = block_split(normal_free)


def get_data(name):
    att = pd.read_parquet(FEAT / f"{name}_features.parquet")
    att = att.sort_values("timestamp", kind="stable").iloc[WARMUP:].reset_index(drop=True)
    a_tr, _, a_te = block_split(att)
    train = pd.concat([a_tr, n_train], ignore_index=True)
    train = train.sample(n=min(len(train), TRAIN_SAMPLE), random_state=0)
    return train, a_te.reset_index(drop=True)


def fit(train, cols):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=-1, verbose=-1)
    return m.fit(train[cols], train.label)


def score(model, test, cols):
    pred = (model.predict_proba(test[cols])[:, 1] >= 0.5).astype(int)
    y = test.label.to_numpy()
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


# %% [markdown]
# ## 1. Feature-group ablation  +  keep models for check 3

# %%
tests, full_models, rows = {}, {}, []
for atk in ATTACKS:
    name = atk.lower()
    train, test = get_data(name)
    tests[atk] = test
    print(f"{atk}: ablation ...")
    full_models[atk] = fit(train, FEATURE_COLS)
    rows.append({"attack": atk, "setup": "all features", **score(full_models[atk], test, FEATURE_COLS)})
    for g, cols in GROUPS.items():
        if not cols:
            continue
        keep = [c for c in FEATURE_COLS if c not in cols]
        rows.append({"attack": atk, "setup": f"without {g}", **score(fit(train, keep), test, keep)})
        rows.append({"attack": atk, "setup": f"only {g}", **score(fit(train, cols), test, cols)})

abl = pd.DataFrame(rows)
order = ["all features"] + [f"{p} {g}" for g in GROUPS for p in ("without", "only")]
for metric in ["f1", "recall"]:
    print(f"\n--- ablation: {metric} (rows = setup, columns = attack) ---")
    print(abl.pivot(index="setup", columns="attack", values=metric)
          .reindex(order)[ATTACKS].round(3).to_string())
abl.to_csv(Path("reports") / "ablation.csv", index=False)


# %% [markdown]
# ## 2. First injected message of each episode

# %%
metrics = pd.read_csv(Path("reports") / "metrics.csv")
print("\n--- first injected message of each test episode ---")
for atk in ATTACKS:
    test = tests[atk]
    thr = metrics[(metrics.attack == atk) & (metrics.test_set == "attack recording")].threshold.iloc[0]
    booster = lgb.Booster(model_file=f"models/lgbm_{atk.lower()}.txt")
    proba = booster.predict(test[FEATURE_COLS])
    y, ts = test.label.to_numpy(), test.timestamp.to_numpy()
    att = np.flatnonzero(y == 1)
    first = att[np.concatenate([[True], np.diff(ts[att]) > EPISODE_GAP])]
    print(f"\n{atk}: {len(first)} episodes, first injected message flagged in "
          f"{(proba[first] >= thr).mean():.0%} of them (threshold {thr:.3f})")
    imp = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLS).nlargest(5).index
    cmp = pd.DataFrame({
        "first msg of episode": test.iloc[first][imp].median(),
        "all injected": test[test.label == 1][imp].median(),
        "regular": test[test.label == 0][imp].median(),
    })
    print(cmp.round(3).to_string())


# %% [markdown]
# ## 3. Cross-attack transfer (F1, threshold 0.5)
# Rows = trained on, columns = tested on.

# %%
cross = pd.DataFrame({te: {tr: score(full_models[tr], tests[te], FEATURE_COLS)["f1"]
                           for tr in ATTACKS} for te in ATTACKS})
print("\n--- cross-attack transfer: F1 ---")
print(cross.round(3).to_string())
