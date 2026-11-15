# Rebuilding the Reddit tiers

Reddit's terms do not permit redistributing Reddit data or tables derived from it, so OpenPopBench releases no Reddit data. It releases three things:
- these scripts;
- one permutation file of integers (`reddit_id_perm.parquet`, on the Hugging Face dataset `Baines/openpopbench-reddit`);
- content checksums.

Together they rebuild the exact v1.0 Reddit tiers from the public source. A rebuild with a different pseudonymization key reproduced every table and query file, with identical content hashes.

## 1. Get the source

Hessel, Tan and Lee (ICWSM 2016), dataset v1.1, metadata only.
- Project page: https://www.cs.cornell.edu/~jhessel/projectPages/redditHRC.html (section "Dataset download").
- It links the 10 GB archive `Hessel_ICWSM_2016.tar` and its readme.

Extract the archive. `rebuild.sh` checks every file against `SOURCE_SHA256.txt` (17,080 files) before it starts.

## 2. Rebuild

```bash
bash scripts/reddit/rebuild.sh /path/to/Hessel_ICWSM_2016 /path/to/reddit_id_perm.parquet /path/to/out 0
export OPB_DATA=/path/to/out/release
```

The script runs the steps used for the paper:
1. Parse the source.
2. Build the full tables. Accounts are numbered with the permutation file.
3. Draw the Forest Fire tiers: S and Mid with p_f = 0.35 and seed 1; M with p_f = 0.7 and seed 1.
4. Sample the queries and the 10% subsample.
5. Derive the newcomer, Task 2a and Task 3 files.
6. Write the release layout.
7. Compare the content hash of every released file with `MANIFEST_RELEASE.tsv`.

It needs about 200 GB of disk and one GPU, and takes one to two hours on 40 cores.

## Why the permutation file is needed

The parser replaces each username with a keyed hash whose key stays private. Without the permutation file, accounts would be numbered in the order of that hash, so any rebuild would number them differently, and the tier sampler would draw different accounts.

The permutation fixes this. It maps each account's position in a key-free canonical order (first appearance in the source by time, type and id) to its v1.0 number. It is a permutation of 0 … N−1 and holds no usernames, ids or times.

## Checking a rebuild

`content_hash.py` hashes table content: rows are sorted and columns serialized in a fixed form. The check therefore does not depend on Parquet writer versions or row order.
- `MANIFEST_TABLES.tsv` lists the intermediate tables.
- `MANIFEST_RELEASE.tsv` lists the files of the release layout.
