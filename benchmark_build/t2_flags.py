"""T2 newcomer flags for every T1 floors row (no model runs needed).

    python t2_flags.py            writes $OPB_WORK/eval/t2/flags_<tier>.parquet and prints newcomer shares

Two definitions, both computed per query (event_row of the floors file):
  unseen_train : the participant has no event in any TRAIN cascade of the tier (the inductive-learning definition)
  first_part   : the participant has no cascade event before the query time t (their first participation anywhere in the tier)
first_part implies unseen_train for validation and test queries whose time is after all training events, so it is the stricter one.
OPB_WORK (output root, default bench/) and OPB_TIERS (JSON {tier: [tables dir, queries parquet]}) override the defaults; the
Task 2a and Task 3 scripts import both, so a rebuild (e.g. scripts/reddit/rebuild.sh) runs them on its own tiers.
"""
import json
import os

import duckdb
import pandas as pd

BB = os.environ.get("OPB_WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
TIERS = {  # tier: (benchmark dir, floors parquet)
    "bluesky_S": (f"{BB}/bluesky/v0.1_ffgrid/pf0.35_s1/tier_S", f"{BB}/eval/ffgrid/floors_pf0.35_s1_S.parquet"),
    "bluesky_Mid": (f"{BB}/bluesky/v0.1_ladderE/pf0.35_s1/tier_Mid", f"{BB}/eval/ladderE/floors_bluesky_Mid.parquet"),
    "bluesky_M": (f"{BB}/bluesky/v0.1_ffgrid/pf0.7_s1/tier_M", f"{BB}/eval/ffgrid/floors_pf0.7_s1_M.parquet"),
    "bluesky_L": (f"{BB}/bluesky/v0.1", f"{BB}/eval/floors_bluesky_L.parquet"),
    "telegram_L": (f"{BB}/telegram/v0.1", f"{BB}/eval/floors_telegram_L.parquet"),
    "reddit_S": (f"{BB}/reddit/v0.1_ladderE/pf0.35_s1/tier_S", f"{BB}/eval/ladderE/floors_reddit_S.parquet"),
    "reddit_Mid": (f"{BB}/reddit/v0.1_ladderE/pf0.35_s1/tier_Mid", f"{BB}/eval/ladderE/floors_reddit_Mid.parquet"),
    "reddit_M": (f"{BB}/reddit/v0.1_ladderE/pf0.7_s1/tier_M", f"{BB}/eval/ladderE/floors_reddit_M.parquet"),
    "reddit_L": (f"{BB}/reddit/v0.1", f"{BB}/eval/floors_reddit_L.parquet"),
}
if os.environ.get("OPB_TIERS"): TIERS = {k: tuple(v) for k, v in json.loads(os.environ["OPB_TIERS"]).items()}
if __name__ == "__main__":
    OUT = f"{BB}/eval/t2"; os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
    rows = []
    for tier, (B, FL) in TIERS.items():
        if not (os.path.exists(FL) and os.path.exists(f"{B}/events.parquet")):
            print(tier, "missing", B if not os.path.exists(f"{B}/events.parquet") else FL); continue
        f = con.execute(f"""
            WITH ev AS (SELECT e.participant_id, e.t, c.split FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)),
                 tr AS (SELECT DISTINCT participant_id FROM ev WHERE split = 'train'),
                 fp AS (SELECT participant_id, min(t) t0 FROM ev GROUP BY participant_id),
                 fl AS (SELECT event_row, participant_id, t, split, in_all_sample, in_novel_sample, novel FROM read_parquet('{FL}'))
            SELECT fl.event_row, fl.split, fl.in_all_sample, fl.in_novel_sample, fl.novel,
                   tr.participant_id IS NULL AS unseen_train, fp.t0 >= fl.t AS first_part
            FROM fl LEFT JOIN tr USING (participant_id) LEFT JOIN fp USING (participant_id) ORDER BY event_row""").df()
        f.to_parquet(f"{OUT}/flags_{tier}.parquet", index=False)
        for sp in ("val", "test"):
            s = f[(f.split == sp) & f.in_all_sample]
            rows.append(dict(tier=tier, split=sp, n_all_sample=len(s), unseen_train=s.unseen_train.mean(), n_unseen=int(s.unseen_train.sum()),
                             first_part=s.first_part.mean(), n_first=int(s.first_part.sum()), novel_among_unseen=s[s.unseen_train].novel.mean(),
                             n_unseen_scored=int(f[(f.split == sp) & f.unseen_train].shape[0])))
        print(tier, "done", flush=True)
    r = pd.DataFrame(rows); print(r.to_string(index=False, float_format="%.3f")); r.to_csv(f"{OUT}/shares.csv", index=False)
