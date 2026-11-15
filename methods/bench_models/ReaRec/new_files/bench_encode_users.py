"""BENCH compatibility step: BaseReader converts user_id to an integer tensor, but the released remap notebooks leave
Amazon user ids as strings. Encode them as consecutive integers over train/valid/test (first-seen order). The models and
runner never read user_id (checked with grep), so this changes no computation."""
import sys
import pandas as pd

d = sys.argv[1]; name = d.rstrip("/").split("/")[-1]
dfs = {k: pd.read_csv(f"{d}/{name}.{k}.remap.csv") for k in ("train", "valid", "test")}
codes = {u: i for i, u in enumerate(pd.unique(pd.concat([dfs[k].user_id for k in dfs])))}
for k, df in dfs.items():
    if df.user_id.dtype == object: df["user_id"] = df.user_id.map(codes); df.to_csv(f"{d}/{name}.{k}.remap.csv", index=False)
print(f"{name}: {len(codes):,} users encoded")
