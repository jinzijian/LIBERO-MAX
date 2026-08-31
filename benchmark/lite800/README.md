# LIBERO-MAX Lite

LIBERO-MAX Lite is the fixed 800-pair evaluation track for rapid policy
integration, ablation, and early comparison. It is a strict subset of the
released Full 8000 track.

The split contains 100 cases from each of the eight online event types. Every
event contains 70 LIBERO-Plus cases and 30 LIBERO-PRO cases, preserving the
70:30 source composition of Full 8000. The split was sampled once with seed
`20260830`, before inspecting policy outcomes, and the same case IDs are used
for every checkpoint.

| Track | Matched pairs | Scored rollouts per checkpoint | Intended use |
| --- | ---: | ---: | --- |
| Lite 800 | 800 | 1,600 | Integration, debugging, ablation, early comparison |
| Full 8000 | 8,000 | 16,000 | Primary benchmark reporting |

The manifest records the benchmark default query interval for schema
compatibility. Model evaluations should apply the same released native query
cadence in Lite and Full. Case membership is independent of query cadence.

Validate the release:

```bash
make validate-lite800
```

Rebuild it deterministically from Full 8000:

```bash
python scripts/build_lite800_split.py
```

`case_index.csv` makes the released membership easy to audit.
`selection_summary.json` records the seed, quotas, and source digest.
