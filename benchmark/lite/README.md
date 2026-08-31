# LIBERO-MAX Lite

LIBERO-MAX Lite is the fixed 400-pair evaluation track for rapid policy
integration, ablation, and early comparison. It is a strict subset of the
released LIBERO-MAX track. We shorten the two track names to Lite and Max in
tables and figures.

The split contains 50 cases from each of the eight online event types. Every
event contains 35 LIBERO-Plus cases and 15 LIBERO-PRO cases, preserving the
70:30 source composition of Max. Lite is an outcome-blind half-sample of the
frozen 800-case cadence pool. The candidate pool uses seed `20260830` and the
Lite selection uses seed `20260831`. The same case IDs apply to every
checkpoint and every query cadence.

| Track | Matched pairs | Scored rollouts per checkpoint | Intended use |
| --- | ---: | ---: | --- |
| Max | 8,000 | 16,000 | Primary benchmark reporting |
| Lite | 400 | 800 | Integration, debugging, ablation, early comparison |

The manifest records the benchmark default query interval for schema
compatibility. Model evaluations should apply the same released native query
cadence in Lite and Max. Case membership is independent of query cadence.

Validate the release:

```bash
make validate-lite
```

Rebuild it deterministically from Max:

```bash
python scripts/build_lite_split.py
```

`case_index.csv` makes the released membership easy to audit.
`selection_summary.json` records the seed, quotas, and source digest.
