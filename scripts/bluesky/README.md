# Rebuilding the Bluesky tiers

The Bluesky tiers are built from the anonymized export of a full-network AT Protocol mirror by Balduf et al. (IMC 2024; https://bsky.leobalduf.com/datasets.html), snapshot of 14 April 2025. Until the export's authors confirm its redistribution terms, OpenPopBench v1.0 releases no Bluesky tables. It releases a rebuild package instead (Hugging Face dataset `Baines/openpopbench-bluesky`):
- `bluesky_id_perm.parquet`, a permutation of integers that fixes account numbering;
- `tier_S_members.parquet`, `tier_Mid_members.parquet`, `tier_M_members.parquet`, the accounts of each sampled tier, as account numbers of the full tier;
- `SOURCE_SHA256.txt`, checksums of the three source files;
- `MANIFEST_RELEASE.tsv`, content hashes of every released table and query file;
- `tiers.json`, the provenance of each tier.

With your own copy of the export, these scripts rebuild the exact v1.0 tables.

## 1. Get the source

Obtain the export from its authors (https://bsky.leobalduf.com/datasets.html) under their terms, and cite their paper. The build reads `follows.parquet`, `posts.parquet` and `reposts.parquet`. `rebuild.sh` checks them against `SOURCE_SHA256.txt` before it starts.

## 2. Rebuild

```bash
bash scripts/bluesky/rebuild.sh /path/to/export /path/to/openpopbench-bluesky /path/to/out 0
export OPB_DATA=/path/to/out/release
```

The script runs the steps used for the paper:
1. Build the full tables (L). Accounts are numbered with the permutation file.
2. Restrict L to the accounts of S, Mid and M, read from the membership files.
3. Sample the queries and the 10% subsample.
4. Derive the newcomer, Task 2a and Task 3 files.
5. Write the release layout.
6. Compare the content hash of every released file with `MANIFEST_RELEASE.tsv`.

It needs about 400 GB of RAM, 300 GB of disk and one GPU, and takes several hours on 64 cores.

## Why the package holds these files

- **Permutation.** The build numbers accounts in the order of DuckDB's `hash()` of the export's account id, which is not guaranteed to be stable across DuckDB versions. The permutation maps each account's position in ascending export id to its v1.0 number. It is a permutation of 0 … N−1 and holds no ids or times.
- **Membership files.** Bluesky S and M were drawn on 2026-10-12 by an earlier version of the Forest Fire sampler, which the released sampler does not reproduce (`tiers.json`). The membership files fix every sampled tier exactly. `ff_tiers.py` with `MEMBERS=<tier>=<file>` restricts L to those accounts with the same code as a sampled tier.

## Checking a rebuild

`content_hash.py` hashes table content: rows are sorted and columns serialized in a fixed form. The check therefore does not depend on Parquet writer versions or row order.
