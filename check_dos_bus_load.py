# %% [markdown]
# # Check: does the DoS flood ADD traffic or DISPLACE legitimate traffic?
#
# Hypothesis (from the Stage 2 feature table, not yet verified):
#   the bus is near capacity, so injected ID 0x000 frames push legitimate
#   frames out instead of raising the total message rate.
#
# Test, per one-second bucket of the DoS attack recording:
#   - "calm"  seconds: no attack message at all
#   - "heavy" seconds: attack messages >= median of all attack-active seconds
# If DISPLACEMENT is true:  total/s stays ~flat, legit/s drops by ~attack/s
# If ADDITIVE is true:      total/s rises by ~attack/s, legit/s stays flat

# %%
from common import *

df = pd.read_parquet(PROC / "dos_attack.parquet")
normal = pd.read_parquet(PROC / "normal_tail.parquet")

sec = (df.timestamp - df.timestamp.iloc[0]).astype(int)
g = df.groupby(sec)
ps = pd.DataFrame({"total": g.size(), "attack": g.label.sum()})
ps["legit"] = ps.total - ps.attack

active = ps.attack > 0
heavy = active & (ps.attack >= ps.attack[active].median())
calm = ~active

nsec = (normal.timestamp - normal.timestamp.iloc[0]).astype(int)
normal_rate = normal.groupby(nsec).size().mean()

# %% [markdown]
# ## 1. Per-second rates

# %%
table = pd.DataFrame({
    "seconds": [calm.sum(), heavy.sum()],
    "total/s": [ps.total[calm].mean(), ps.total[heavy].mean()],
    "legit/s": [ps.legit[calm].mean(), ps.legit[heavy].mean()],
    "attack/s": [ps.attack[calm].mean(), ps.attack[heavy].mean()],
}, index=["calm", "heavy attack"]).round(1)
print(table)
print(f"\nnormal-only tail reference: {normal_rate:.1f} msgs/s")

lost = table.loc["calm", "legit/s"] - table.loc["heavy attack", "legit/s"]
injected = table.loc["heavy attack", "attack/s"]
print(f"\nlegit msgs lost per second : {lost:.1f}")
print(f"attack msgs injected/second: {injected:.1f}")
print(f"displacement fraction      : {lost / injected:.2f}   (1.0 = pure displacement, 0.0 = purely additive)")

# %% [markdown]
# ## 2. Which legitimate IDs lose messages?
# CAN arbitration is by ID: lower ID = higher priority. If the flood really
# displaces traffic, low-priority (high-value) IDs should lose the most.

# %%
legit = df[df.label == 0]
cnt = (legit.groupby([sec[legit.index], "can_id"]).size()
       .unstack(fill_value=0).reindex(ps.index, fill_value=0))
rate_calm = cnt[calm.values].mean()
rate_heavy = cnt[heavy.values].mean()
per_id = pd.DataFrame({"calm/s": rate_calm, "heavy/s": rate_heavy})
per_id = per_id[per_id["calm/s"] >= 5]
per_id["kept"] = per_id["heavy/s"] / per_id["calm/s"]
per_id.index = [f"0x{i:03x}" for i in per_id.index]
print(per_id.round(2).sort_index().to_string())

ids = pd.Series([int(i, 16) for i in per_id.index], index=per_id.index)
print(f"\nSpearman(can_id, share kept) = {per_id.kept.corr(ids, method='spearman'):.2f}"
      "  (strongly negative = low-priority IDs lose most = arbitration)")

# %%
