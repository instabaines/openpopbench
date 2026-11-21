# Datasheet: OpenPopBench Bluesky (v1.0, released as a rebuild package)

Format: Gebru et al., *Datasheets for Datasets* (CACM 2021). Numbers from the paper (Table 1 and Appendix A).

## Motivation

- **Purpose.** To evaluate next-participant ranking (Task 1), newcomer prediction (Task 2) and event timing (Task 3) for spreading content at platform scale, under full-candidate ranking and chronological splits. Existing diffusion datasets hold tens of thousands of accounts, use random splits and sampled negatives, and contain few accounts that first appear after training.
- **Creators.** The OpenPopBench authors (see the paper). The source export was created by Balduf et al. (IMC 2024).
- **Funding.** See the paper's acknowledgements.

## Composition

- **Instances.** Four tables per tier:
  - `events`: cascade, participant, time (Unix seconds) and position;
  - `cascades`: origin account, start time, split;
  - `edges`: timestamped follow ties, oriented from the followed account to the follower;
  - `participants`: first-seen time.

  A cascade is a post that was reposted; an event is the first repost of that post by an account within 72 hours of its creation.
- **Size.** The full tier (L) has 13,725,475 candidate accounts, 16.1M cascades, 132.9M events and 922M edges. Three sampled tiers have 50,000 accounts (S), 150,001 (Mid) and 500,004 (M).
- **Sampling.** L is every reposted post whose creation time falls in the split windows, from a full-network snapshot. S, Mid and M are Forest Fire samples of L, each drawn separately (S is not nested in Mid). S and M were drawn by an earlier version of the sampler, so the released tables, not the sampler, are their reference (`tiers.json` in the Hugging Face dataset). The sampler's burning probability was chosen on validation data by pre-registered rules (paper, Appendix A). Sampled tiers keep fewer network neighbours than L (paper, Appendix F).
- **Labels.** The tasks' targets are derived from the event stream: the next participant, whether that participant is a newcomer (first seen after the training window), and the gap to the next event. Query files list the evaluated events: 20,000 per slice and split, plus a 2,000-query subsample for pairwise methods.
- **Missing information.** Posts that were never reposted are absent. Some follow edges have no plausible creation time; their `t_created` is null (267 of 922M edges in L). Accounts deleted before the snapshot are absent from the source.
- **Confidentiality and sensitive content.** The tables contain no text, media, profile fields, handles, DIDs or record keys. Accounts are integers that we assigned over the source export's own random integers (`did_id`); only the source authors hold the key to DIDs. Exact timestamps are kept because every task depends on event order. Timestamps carry a residual re-identification risk, which the access agreement addresses.
- **Relation to people.** Accounts are mostly individual people. No demographic or other attribute is included or inferred.

## Collection process

- **Source.** The anonymized export of a full-network AT Protocol mirror (Balduf et al., IMC 2024; https://bsky.leobalduf.com/datasets.html), snapshot of 14 April 2025. We collected no data from the network ourselves.
- **Time frame.** Cascade start times:
  - training: 3–28 March 2025;
  - validation: 31 March–4 April 2025;
  - test: 7–11 April 2025.

  The stream ends 14 April 2025, so every test cascade is observed for its full 72-hour horizon.
- **Consent.** The data are public posts, collected and pseudonymized by the source authors; account holders did not consent individually to this use. Our use is secondary analysis of pseudonymized metadata, with no text and no individual-level outputs (paper, Section 9).

## Preprocessing

- **Conversion.** Reposts become first-participation events, keeping the first repost per (post, account). Reposts made before the post's own creation time are dropped (78,826 records). Positions are contiguous from 0 (the origin). Account and cascade identifiers are reassigned as dense integers.
- **Integrity checks.** There are 23 checks on L and 19 on each sampled tier, all passing. They include exact reconciliation of 1,000 random cascades and 100 random edges with the source export (`benchmark_build/check_integrity.py`; paper, Appendix A).
- **Leakage test.** Every pipeline gives identical outputs for queries before a cut time when all data after the cut is deleted (paper, Section 6.1).
- **Raw data.** The source export is not redistributed by us. Our build scripts rebuild the tables from it.

## Uses

- **Intended uses.**
  - Benchmarking the three tasks.
  - Studying the scalability of diffusion, temporal-graph, point-process and sequential-recommendation methods.
  - Studying newcomer (cold-start) prediction.
- **Not intended for:**
  - re-identifying accounts or linking them to the live network;
  - profiling individuals;
  - any decision about an individual.

  The tasks are aggregate prediction tasks.
- **Known limits.**
  - The window is about six weeks in early 2025.
  - Sampled tiers are biased toward denser parts of the network, so rankings on S transfer to larger tiers with τ = 0.52–0.78 on all events.
  - Results are tied to v1.0.

## Distribution

- **v1.0: rebuild package (no tables).** Until the export's authors confirm its redistribution terms, no Bluesky tables are hosted. A Hugging Face dataset (with a DOI) holds:
  - a permutation of integers that fixes account numbering (`bluesky_id_perm.parquet`);
  - the accounts of each sampled tier (`tier_{S,Mid,M}_members.parquet`);
  - checksums of the three source files and content hashes of every table and query file;
  - `tiers.json`.

  With a copy of the source export, `scripts/bluesky/rebuild.sh` in the code repository rebuilds the exact tables, query files and Task 2 flags.
- **Later: hosted tables.** Once the terms are confirmed, the tables will be hosted with gated download: a click-through agreement with automatic approval that permits research use only and prohibits re-identification and linkage to the live network.
- **Terms.** The terms of the source export, which ask users to cite Balduf et al. (IMC 2024). Cite the OpenPopBench paper as well.

## Maintenance

- **Maintainer.** The OpenPopBench authors. Contact: raamure@ualr.edu.
- **Updates.**
  - v1.0 is frozen to the paper.
  - Every six months, the tables are rebuilt from the newest source export, so that removals made upstream carry over. Accounts named in takedown requests are also dropped.
  - Each version has a changelog and leaderboards re-scored from the maintainers' checkpoints.
- **Takedown.** Requests go to the contact address above and are applied in the next scheduled release.
- **Older versions.** They stay available for reproducing published numbers, unless a takedown requires their withdrawal.
