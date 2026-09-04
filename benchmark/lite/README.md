# LIBERO-MAX Lite

LIBERO-MAX Lite is the fixed 800-pair evaluation track for rapid policy
integration, ablation, and early comparison. It is a strict subset of the
released LIBERO-MAX benchmark and uses the same paired protocol and event
taxonomy.

The split contains 100 cases from each of the eight online event types. Every
event contains 70 LIBERO-Plus cases and 30 LIBERO-PRO cases, preserving the
70:30 source composition of LIBERO-MAX. A persistent pseudorandom generator with seed
`20260830` selects the outcome-blind subset before any policy result is
inspected. These are also the fixed case IDs used by the cadence analysis. The
same case IDs apply to every checkpoint and every query cadence.

| Benchmark | Matched pairs | Scored rollouts per checkpoint | Intended use |
| --- | ---: | ---: | --- |
| LIBERO-MAX | 8,000 | 16,000 | Primary benchmark reporting |
| LIBERO-MAX Lite | 800 | 1,600 | Integration, debugging, ablation, early comparison |

The manifest records the benchmark default query interval for schema
compatibility. Model evaluations should apply the same released native query
cadence in LIBERO-MAX Lite and LIBERO-MAX. Case membership is independent of
query cadence.

Validate the release:

```bash
make validate-lite
```

Rebuild it deterministically from LIBERO-MAX:

```bash
python scripts/build_lite_split.py
```

## Evaluate a checkpoint on LIBERO-MAX Lite

Use the same checkpoint and native serving configuration that would be used on
LIBERO-MAX. LIBERO-MAX Lite changes case membership only; it does not change the
action horizon, query interval, decoding steps, or model seed. From the
repository root, the generic scheduler can run any compatible shard adapter on
every visible GPU. For example:

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
Pass `--gpus 0,2` to use an explicit subset. A complete LIBERO-MAX Lite run contains 800
terminal pair records and scores 1,600 rollouts: one Base and one Dynamic rollout
for every released case ID. Failed or missing cases stay in the 800-pair
denominator.

After validating an integration on LIBERO-MAX Lite, evaluate LIBERO-MAX by
replacing the manifest with `benchmark/max8000/libero_max_8000.json` and using a
separate output root. Do not change the checkpoint or serving configuration
between benchmarks.

`case_index.csv` makes the released membership easy to audit.
`selection_summary.json` records the seed, quotas, and source digest.
