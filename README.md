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
- Retain every frozen benchmark case even when a policy fails before reaching
  the intervention trigger; report it as a pre-intervention policy failure.
- Report full-denominator end-to-end success, trigger coverage, and
  trigger-conditioned adaptation separately.
- Treat simulator, dependency, and missing-trace errors as infrastructure
  gaps that block finalization, not as model failures.
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
case; each GPU runs one matched pair at a time. Aggregation retains terminal
`trigger_unreached` cases in end-to-end scoring, while missing traces,
simulator errors, and mismatched pairs block finalization rather than being
silently converted into model failures.

## LIBERO-MAX-8000 release

Version 3.0.0 adds a source-locked **MAX-PRO-Hard-2400** subset to the released
Base-5600:

- **MAX-Base-5600:** the existing balanced LIBERO-Plus dynamic track;
- **MAX-PRO-Hard-2400:** 10 LIBERO-PRO substrate categories x 8 MAX changes x
  2 frozen draws x 15 pairs per joint cell;
- **LIBERO-MAX-8000:** 8,000 matched pairs, or 16,000 rollouts per model.

The PRO control and intervention arms use the same perturbed BDDL, init state,
instruction, policy seed, and pre-event action chunks. Their only paired
difference is whether the MAX event fires after execution begins. Results must
report Base-5600 and PRO-Hard-2400 separately before any combined aggregate.
Config-driven PRO perturbations are applied after frozen-state restoration and
on every policy observation; topology-extended occlusion tasks use a
joint-name state adapter rather than padding the MuJoCo vector by position.

The candidate is pinned to the public LIBERO-PRO dataset revision
`c86fc3b8293185a6f373677018ff3e37f8391602`. It contains 400 distinct PRO task
variants; the 2,400 pairs cross those variants with different dynamic events,
draws, and frozen init states. Upstream `runtime_object_move` is excluded
because it already changes the world mid-execution, and the environment track
is deferred because complete BDDL/init artifacts are absent from the pinned
revision.

The frozen release lives in [`benchmark/max8000`](benchmark/max8000), with
construction provenance retained under
[`benchmark/max8000_candidate`](benchmark/max8000_candidate). Its real-MuJoCo
preflight passes **2,400/2,400** configurations while preserving every balanced
cell and strong draw. The source-locked PRO-aware runtime is publicly
retrievable at the recorded commit, and the version 3.0.0 freeze checksums are
verified in CI. See [`docs/MAX_PRO_HARD_DESIGN.md`](docs/MAX_PRO_HARD_DESIGN.md)
for dataset construction and
[`docs/MAX8000_EXPERIMENT_MATRIX.md`](docs/MAX8000_EXPERIMENT_MATRIX.md) for
the frozen paper evaluation and publication gates.

## Released MAX-Base-5600 on LIBERO-Plus

The currently released physical benchmark is **LIBERO-MAX-5600**: 5,600 matched
control/intervention pairs, or 11,200 rollouts per model. It is a deterministic
balanced subset of the 10,030 LIBERO-Plus tasks:

- 7 Plus generalization categories x 800 pairs;
- 8 mid-execution changes x 700 pairs;
- 2 frozen intervention draws x 350 pairs per change;
- exactly 50 pairs in every category/change/draw cell.

The eight changes are lighting, camera pose/FOV, visual theme, sensor
corruption, target relocation, receptacle relocation, a five-object distractor
burst, and obstacle insertion. Relocation directions and clutter placements
are frozen in the manifest rather than sampled again at evaluation time. The
1,400-case development Core seeds the selection but is not a second public
benchmark profile. Any Core candidate that fails the final intervention
contract is replaced under the same balance constraints.

### Scoring and trigger coverage

The 5,600-case manifest is model-independent. A case is never removed because
a particular policy fails to approach the trigger entity. Such an episode is
retained as `trigger_unreached` / `pre_intervention_failure`: it contributes a
failure to the full end-to-end denominator but is not treated as evidence
about post-change adaptation.

Every physical-track result therefore reports four distinct views:

| View | Denominator | Meaning |
| --- | ---: | --- |
| End-to-end control/change | All planned cases in the reported track | Overall policy capability, including failures before intervention |
| Trigger coverage | Triggered cases / 5,600 | How often the policy reached a state in which adaptation could be tested |
| Response-query coverage | Cases with a paired post-event policy query / 5,600 | How often the event occurred early enough for the fixed-commitment policy to observe it |
| Response-conditioned adaptation | Valid post-event-query pairs only | Behavior change after the intervention could enter a policy query |

An intervention that fires after the last policy query remains a
full-denominator model/horizon outcome. It is not repaired, dropped, or
assigned a fabricated post-event action difference.

Infrastructure failures remain missing until repaired. They are neither
dropped from the benchmark nor charged to the policy. Consequently,
planned-denominator accuracies are withheld until every control and
intervention arm has a terminal outcome.

Every released case must pass real-MuJoCo post-intervention geometry, support,
collision, and visibility checks. Failed candidate placements are replaced
within the same balanced cell; they are never counted as policy failures. The
separate 96-pair Intent track covers instruction target/receptacle updates and
task cancellation. See [`docs/MAX_HARD_DESIGN.md`](docs/MAX_HARD_DESIGN.md)
for the construction, frozen randomness, feasibility, and reporting contract.

On the configured LIBERO-Plus host, run a frozen manifest preflight and Cosmos
paired evaluation with one persistent worker per physical GPU:

```bash
GPUS=0,1,2,3,4,5,6,7 bash scripts/run_max_hard_preflight.sh \
  benchmark/max5600/libero_max_5600.json \
  artifacts/max5600/preflight_recheck

GPUS=0,1,2,3,4,5,6,7 bash scripts/run_max_hard_cosmos.sh \
  benchmark/max5600/libero_max_5600.json \
  artifacts/max5600/cosmos_policy

GPUS=0,1,2,3,4,5,6,7 bash scripts/run_openpi_persistent_benchmark.sh \
  benchmark/max5600/libero_max_5600.json \
  artifacts/max5600/pi05_libero
```

<!-- PAPER_RESULTS_START -->
## Paper-scale results

The evidence-gated MAX-8000 and frozen cross-model evaluations are currently
running. Final tables, figures, human-feasibility review queue, and verified
simulator media are withheld until every planned arm has a terminal outcome
and all infrastructure gaps have been repaired.
<!-- PAPER_RESULTS_END -->

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

## Legacy v1 results

The project also retains the original six-type v1 release for reproducibility.
It has an executable Cosmos physical-change pilot plus a frozen
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
