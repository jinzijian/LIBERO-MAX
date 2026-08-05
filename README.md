# LIBERO-MAX

**A Benchmark for Mid-Execution Adaptation to Exogenous Changes**

LIBERO-MAX evaluates whether vision-language-action (VLA) models respond
appropriately when the world changes after task execution has already begun.

The name expands to:

> **LIBERO-MAX: Benchmarking Mid-Execution Adaptation to eXogenous Changes in
> Vision-Language-Action Models**

## Core question

Can a VLA model detect an externally introduced change, revise its behavior,
and still produce the appropriate outcome without causing a new failure?

"Appropriate outcome" is deliberately broader than the original task
predicate. The physical track remains feasible and requires task completion;
the intent track requires the revised goal or a measured safe stop after
cancellation.

## Benchmark scope

LIBERO-MAX covers changes introduced during an episode, including:

- lighting changes and camera displacement;
- reachable target-object or receptacle relocation;
- newly introduced obstacles;
- sudden bursts of multiple visually confusing distractors;
- user instruction modification or cancellation.

Every v1 physical intervention is preflighted to remain reachable and
finishable. Explicit infeasibility awareness is reserved for a future track
with a model-agnostic abstention interface.

A change is **exogenous** when it is introduced by the environment, benchmark,
or user rather than being caused solely by the robot's own execution error.

## Evaluation principles

- Compare matched no-change and changed episodes using the same task, seed, and
  initial state whenever possible.
- Record the intervention time and retain both pre-change and post-change
  trajectories.
- Report complete paired coverage, not only aggregate success rates.
- Separate gains, preserved successes, preserved failures, and regressions.
- Score safe stopping and correct refusal explicitly for cancelled or
  infeasible tasks.
- Break results down by change family, timing, severity, and response mode.

## Quick start

No third-party dependency is required for validation, reporting, or tests.

```bash
# Validate the pilot scenario suite.
PYTHONPATH=src python3 -m libero_max validate examples/scenarios/pilot.json

# Summarize matched pilot outcomes against the planned scenarios.
PYTHONPATH=src python3 -m libero_max summarize \
  examples/results/pilot_results.jsonl \
  --scenarios examples/scenarios/pilot.json

# Run the test suite.
make test

# Validate the executable Cosmos physical-pilot manifest.
make validate-physical-manifest
```

The simulator backend is optional and imports NumPy only when used. A real
LIBERO camera-intervention smoke can be run inside an existing LIBERO runtime:

```bash
PYTHONPATH=src:$LIBERO_REPO python scripts/smoke_libero_runtime.py \
  --suite libero_goal \
  --task-id 0 \
  --scenario-id obs_camera_shift_001
```

For Cosmos Policy, use the paired launcher on a free GPU. It runs matched
frozen control and intervention arms with the same task, initial state, and
policy seed:

```bash
GPU_ID=0 OUTPUT_ROOT=artifacts/cosmos_paired_smoke \
  bash scripts/run_cosmos_paired_smoke.sh
```

Run the manifest-driven, resume-safe physical pilot on one or more free GPUs:

```bash
# Verify every setup/change against a real MuJoCo environment without loading
# the policy checkpoint.
PYTHONPATH=src python3 scripts/preflight_manifest_interventions.py \
  examples/manifests/cosmos_physical_pilot_v0.1.json

# Calibrate a target-distance trigger from a previously successful control
# action log before freezing the threshold.
PYTHONPATH=src python3 scripts/calibrate_proximity_from_console.py \
  artifacts/cosmos_paired_smoke/control/console.log

# Rebuild, calibrate, preflight, coverage-audit, and freeze the complete v1
# Core/Full test sets. Relocation directions are calibrated before manifests
# are generated; the pipeline never samples them during evaluation.
bash scripts/run_v1_dataset_build.sh

PYTHONPATH=src python3 scripts/run_cosmos_benchmark.py \
  examples/manifests/cosmos_physical_pilot_v0.1.json \
  --output-root artifacts/cosmos_physical_pilot_v0.1 \
  --gpus 0,1 --resume

PYTHONPATH=src python3 scripts/aggregate_cosmos_benchmark.py \
  artifacts/cosmos_physical_pilot_v0.1
```

The batch runner fixes task, initial-state index, policy seed, and scenario per
case; each GPU runs one matched pair at a time. Aggregation fails on missing or
mismatched pairs rather than silently reporting partial metrics.

## MAX-Hard on LIBERO-Plus

The next benchmark profile composes all 10,030 installed LIBERO-Plus tasks with
one of eight mid-execution events: lighting, camera pose/FOV, visual theme,
sensor corruption, target relocation, receptacle relocation, distractor burst,
or obstacle insertion. Its candidate Core contains 1,400 matched pairs using
an exact 7 Plus categories x 5 difficulty levels x 40 design; candidate Full
contains one matched pair for every Plus task. Core is balanced at 175 pairs
per dynamic event and is an exact Full subset.

The candidate is being promoted only through resolved real-MuJoCo preflight;
failed physical placements are replaced within the same Core stratum rather
than counted as model failures. Complete policy rollouts are required before
claiming that MAX-Hard is empirically harder than Plus. See
[`docs/MAX_HARD_DESIGN.md`](docs/MAX_HARD_DESIGN.md) for the construction,
randomness, feasibility, and reporting contract.

On the configured LIBERO-Plus host, run a frozen manifest preflight and Cosmos
paired evaluation with one persistent worker per physical GPU:

```bash
GPUS=0,1,2,3,4,5,6,7 bash scripts/run_max_hard_preflight.sh \
  artifacts/max_hard/core.final.json \
  artifacts/max_hard/core_preflight_final

PYTHONPATH=src python scripts/run_cosmos_persistent_benchmark.py \
  artifacts/max_hard/core.final.json \
  --output-root artifacts/max_hard/cosmos_core \
  --gpus 0,1,2,3,4,5,6,7 \
  --t5-embeddings /path/to/libero_plus_t5_embeddings.pkl
```

## Repository layout

```text
LIBERO-MAX/
├── README.md
├── docs/BENCHMARK_SPEC.md
├── schemas/
│   ├── manifest.schema.json
│   ├── result.schema.json
│   └── scenario.schema.json
├── src/libero_max/
│   ├── cli.py
│   ├── cosmos_integration.py
│   ├── libero_backend.py
│   ├── results.py
│   ├── runtime.py
│   └── scenario.py
├── examples/
│   ├── results/pilot_results.jsonl
│   └── scenarios/pilot.json
└── tests/
```

The project now has an executable Cosmos physical-change pilot plus a frozen
six-type v1 physical test-set release. Core contains 1,335 matched pairs / 2,670
episodes; Full contains 4,005 matched pairs / 8,010 episodes and crosses three
policy seeds with every retained intervention configuration. All 1,335 unique
physical scenarios passed the real-MuJoCo preflight; the release records the 62
candidate configurations excluded by the feasibility filter.

The v1 paper evaluation is complete for Cosmos Policy Predict2-2B and
pi0.5-LIBERO: 1,335/1,335 Physical Core pairs and 96/96 Intent Core pairs per
model, all with 100% trigger coverage. The release also includes a balanced
180-pair Cosmos query-interval ablation and a matched explicit-notification
ablation. In total, the reported experiments comprise 3,222 independently
executed matched pairs / 6,444 rollouts. Aggregate tables, paired records, and
the result contract are published in [`results/v1`](results/v1/README.md); the
paper-ready write-up is in [`paper/main.md`](paper/main.md).

The infeasibility track remains outside v1 because it still requires a
model-agnostic abstention interface. See
[`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) for the paper-scale roadmap and claim
boundaries. Pilot intervention and proximity-threshold evidence is recorded in
[`docs/PILOT_CALIBRATION.md`](docs/PILOT_CALIBRATION.md).
The current v1 sampling proposal is in
[`docs/BENCHMARK_V1_DESIGN.md`](docs/BENCHMARK_V1_DESIGN.md).
The deterministic build, validation, and checksum contract is documented in
[`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md).

## Working title

**LIBERO-MAX: Do Vision-Language-Action Models Adapt When the World Changes
Mid-Execution?**
