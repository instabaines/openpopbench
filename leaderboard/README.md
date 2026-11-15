# Leaderboard

`v1.0/` holds one folder per method, with `meta.json` (method, code, training protocol, track) and `results.json` (every metric on every tier where the method ran, written by `opb evaluate`). Render the tables with:

```bash
opb leaderboard leaderboard/v1.0 --out leaderboard/v1.0/LEADERBOARD.md
```

The entries of v1.0 are the results of the paper and are all verified: they were run through the harness by the maintainers. Their rank files are in the Hugging Face datasets, under `results/v1.0/`.

**To add an entry**, open a pull request with your submission folder under `submissions/` (see the README, *Submitting a result*). Include the rank files, or a link to them if they are larger than a few MB. We evaluate it and move it into the current version's folder. Verified entries need your adapter and checkpoint; we re-run them through the harness, including the leakage test.
