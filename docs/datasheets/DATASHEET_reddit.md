# Datasheet: OpenPopBench Reddit (v1.0, released as build scripts)

Format: Gebru et al., *Datasheets for Datasets* (CACM 2021). Numbers from the paper (Table 1 and Appendix A).

## Motivation

- **Purpose.** The same three tasks as the other OpenPopBench datasets. This dataset covers a platform where participation mostly does not repeat earlier ties: 85–94% of test comments come from users with no earlier tie to the submitter.
- **Creators.** The OpenPopBench authors. The source is the comment-thread metadata of Hessel, Tan and Lee (ICWSM 2016), v1.1.

## Composition

- **Instances.** The four tables:
  - `events`: submission, user, time, position;
  - `cascades`: submitter, start time, split;
  - `edges`: timestamped at the first time one user replied to another;
  - `participants`: first-seen time.

  A cascade is a submission; an event is a user's first comment on it within 72 hours.
- **Size.**
  - L: 6,879,641 candidate users, 1.21M cascades, 13.1M events, 168M edges.
  - Sampled tiers: S with 50,003 users, Mid with 150,000, M with 500,002.
- **Missing information.** Submissions whose submitter was deleted (832,130 of 3.30M, 25%) are dropped, a possible bias.
- **Sensitive content.** The tables contain no text, usernames or Reddit IDs.

## Distribution: no Reddit data is redistributed

Reddit's current terms do not permit redistribution of Reddit data or derived datasets. We therefore release:
- the build scripts (parser, builder, tier sampler with seeds, query and flag builders);
- `reddit_id_perm.parquet`, a permutation of the integers 0 … N−1 that fixes participant numbering (no usernames, IDs or times);
- SHA-256 checksums of the source files and content hashes of every output table.

With the public source archive (https://www.cs.cornell.edu/~jhessel/projectPages/redditHRC.html), these rebuild content-identical tables. A rebuild with a different pseudonymization key matched every hash (`scripts/reddit/README.md`). No checkpoints trained on Reddit are released.

## Collection process

- **Source.** Hessel et al. v1.1 (Pushshift-derived metadata), six weeks from 3 December 2012 to 14 January 2013.
- **Consent.** The data are public comments, collected by the source authors; account holders did not consent individually to this use. Our use is secondary analysis of metadata, with no text and no individual-level outputs, and no Reddit data is redistributed (paper, Section 9).
- **Time frame.** Cascade start times:
  - training: 3–28 December 2012;
  - validation: 31 December 2012–4 January 2013;
  - test: 7–11 January 2013.

  The training window includes the Christmas and New Year holidays.

## Preprocessing

- 1,003,652,089 lines parsed, 0 skipped.
- All 22 integrity checks pass, including a recount of 1,000 cascades from the parsed lines.
- The leakage test passes (paper, Section 6.1).

## Uses

- **Intended uses.** Benchmarking the three tasks where spread reaches users without earlier ties, the hardest regime for every method.
- **Not intended for** identifying users or any individual-level decision.
- **Known limits.** The data are from 2012–2013.

## Maintenance

- Scripts are maintained in the code repository. A change of the public source would be recorded in a new version with new checksums.
- The contact address is raamure@ualr.edu.
