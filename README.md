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

"Appropriate outcome" is deliberately broader than task completion. Depending
on the change, the correct response may be to replan, ask for clarification,
stop safely, acknowledge cancellation, or report that the task is no longer
feasible.

## Benchmark scope

LIBERO-MAX covers changes introduced during an episode, including:

- lighting changes and camera displacement;
- target-object or receptacle relocation and removal;
- newly introduced obstacles;
- sudden bursts of multiple visually confusing distractors;
- user instruction modification or cancellation;
- transitions from feasible to infeasible tasks.

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

The project now has an executable Cosmos physical-change pilot. The next
milestone is severity calibration and multi-task / multi-seed expansion; intent
and infeasibility tracks still require response-aware scorers. See
[`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) for the paper-scale roadmap and claim
boundaries. Pilot intervention and proximity-threshold evidence is recorded in
[`docs/PILOT_CALIBRATION.md`](docs/PILOT_CALIBRATION.md).

## Working title

**LIBERO-MAX: Do Vision-Language-Action Models Adapt When the World Changes
Mid-Execution?**
