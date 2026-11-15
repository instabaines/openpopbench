# Datasheet: OpenPopBench Telegram (v1.0)

Format: Gebru et al., *Datasheets for Datasets* (CACM 2021). Numbers from the paper (Table 1 and Appendix A).

## Motivation

- **Purpose.** The same three tasks as the other OpenPopBench datasets. This dataset covers a platform where participation mostly repeats earlier ties: 96% of test forwards come from a chat with an earlier tie to the source.
- **Creators.** The OpenPopBench authors. The source corpus is the public US 2024 election Telegram corpus (Harvard Dataverse, doi:10.7910/DVN/1M5KHX).

## Composition

- **Instances.** Participants are chats (channels and groups), not individual users. The four tables:
  - `events`: cascade, participant chat, time, position;
  - `cascades`: source chat, start time, split;
  - `edges`: chat a → chat b, timestamped at the first forward from a seen in b;
  - `participants`: first-seen time. It lists all 313,047 chats seen in the corpus; the 42,983 crawled chats (`is_crawled`) are the candidates, and the others appear only as sources of forwarded posts.

  A cascade is an original post; an event is its first forward into another chat within 7 days.
- **Size.** One tier (L) with 42,983 candidate chats (the crawled chats, whose complete histories are observed), 7.38M cascades, 19.1M events and 2.69M edges.
- **Labels.** Derived from the stream (next participant, newcomer flag, gap to the next event): query files with 20,000 queries per slice and split, plus the Task 2 flags.
- **Missing information.** Forwards into chats outside the crawl are not observed as events. Candidates are restricted to crawled chats.
- **Sensitive content.** The tables contain no message text, media, chat names or Telegram IDs; chats are dense integers. The topic of the source corpus (the 2024 US election) is not encoded in the tables.

## Collection process

- **Source.** The public Dataverse corpus (CC BY 4.0), converted with its forward metadata. We crawled no data ourselves.
- **Consent.** The data are messages from public channels and groups, collected by the corpus authors; channel operators did not consent individually to this use. Our use is secondary analysis of forward metadata, with no text and no individual-level outputs (paper, Section 9).
- **Time frame.** Cascade start times:
  - training: 1 July–24 September 2024;
  - validation: 1–20 October 2024;
  - test: 27 October–23 November 2024.

## Preprocessing

- **Conversion.** One event per (post, forwarding chat), keeping the first forward. Positions are contiguous from 0. Identifiers are dense integers; the map to Telegram IDs is kept private.
- **Integrity checks.** All 22 checks pass, including exact reconciliation of 1,000 random cascades with the corpus. The leakage test passes (paper, Section 6.1).

## Uses

- **Intended uses.** Benchmarking the three tasks, especially at high repeat behaviour, where memorization heuristics are strong.
- **Not intended for** identifying chat operators or any individual-level decision.
- **Known limits.**
  - One election period.
  - Chats are the unit of analysis.
  - The test window includes election day, so drift relative to training is expected.

## Distribution

- Open download on Hugging Face under CC BY 4.0, the source licence. Cite the source corpus and the OpenPopBench paper.
- Includes the tables, query files, Task 2 flags, checksums, Croissant metadata and the v1.0 result files. Each version has a DOI.

## Maintenance

- Versioned releases with changelogs; v1.0 is frozen to the paper.
- The contact address is raamure@ualr.edu. Takedown requests are applied in the next scheduled release.
