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

## Evaluate a checkpoint on Lite

Use the same checkpoint and native serving configuration that would be used on
Max. Lite changes case membership only; it does not change the action horizon,
query interval, decoding steps, or model seed. From the repository root, the
generic scheduler can run any compatible shard adapter on every visible GPU.
For example:

```bash
python scripts/run_dynamic_benchmark.py \
  benchmark/lite/libero_max_lite.json \
  --runner scripts/run_xvla_persistent_shard.py \
  --output-root artifacts/xvla-lite \
  -- \
  --lerobot-root /path/to/lerobot \
  --checkpoint /path/to/xvla-libero-checkpoint \
  --query-interval 30
```

The scheduler discovers all GPUs visible through `CUDA_VISIBLE_DEVICES`,
assigns suite shards dynamically, and retries incomplete shards with `--resume`.
Pass `--gpus 0,2` to use an explicit subset. A complete Lite run contains 400
terminal pair records and scores 800 rollouts: one Base and one Dynamic rollout
for every released case ID. Failed or missing cases stay in the 400-pair
denominator.

After validating an integration on Lite, evaluate Max by replacing the manifest
with `benchmark/max8000/libero_max_8000.json` and using a separate output root.
Do not change the checkpoint or serving configuration between tracks.

`case_index.csv` makes the released membership easy to audit.
`selection_summary.json` records the seed, quotas, and source digest.
