# LIBERO-MAX v1 Benchmark Design

Status: v1.0.0 physical test sets frozen after deterministic relocation
calibration, two-pass real-MuJoCo preflight, feasibility filtering, and release
audit. Cosmos rollout and paper-result validation are still pending.

## Six-type physical main track

The installed catalog contains 40 tasks across LIBERO Spatial, Object, Goal,
and LIBERO-10. Eligibility is defined per intervention rather than forcing an
invalid change into every scene.

| Change type | Eligible tasks | What changes mid-execution |
| --- | ---: | --- |
| Illumination switch | 40 | lights dim sharply or a dim scene lights up |
| Camera shift | 40 | agent-view camera translates and yaws |
| Target relocation | 31 | a target on a floor/table workspace shifts 6 or 12 cm |
| Receptacle relocation | 26 | a movable destination shifts 6 or 12 cm |
| Distractor burst | 12 | five task-native non-target objects appear together |
| Obstacle insertion | 28 | a task-native object appears in the current approach corridor |
| **Task-type cells** | **177** |  |

The catalog initially admitted 33 target-relocation and 27
receptacle-relocation task cells. Calibration retained 31 and 26 respectively:
one fixed direction had to pass both 6 cm and 12 cm in all three initial states.
Receptacle relocation remains restricted to movable destinations on a planar
workspace; the benchmark does not translate a fixed cabinet, stove, or rack to
inflate coverage. The feasibility filter also removed all obstacle draws from
11 task cells while retaining the obstacle family on 28 tasks.

## Explicit intervention randomness

Policy randomness and intervention variation are separate fields. A stable
SHA-256-derived intervention seed is computed from:

```text
(sampler version, suite, task, initial-state index, change type, draw id)
```

The generator resolves every sampled value into the manifest. Re-running a
case therefore cannot silently choose a new direction, distractor, or pose.

| Change type | Pre-resolved variation across draw IDs 0/1/2 |
| --- | --- |
| Illumination | deterministic normal-to-dim scales 0.55/0.30, or a 0.30-to-normal switch; permanent blackout is excluded |
| Camera | three fixed global viewpoint shifts at 2/4/6 cm with yaw +5/-10/+15 degrees |
| Target relocation | one offline-calibrated direction per task; deterministic 6 and 12 cm tiers along the same ray |
| Receptacle relocation | one offline-calibrated direction per task; deterministic 6 and 12 cm tiers along the same ray |
| Distractor burst | task-native identity subset/order and five angularly separated candidate placements around the target |
| Obstacle insertion | task-native obstacle identity, path fraction in [0.35, 0.65], and signed lateral offset |

The semantic trigger remains fixed at 18 cm in the main table. Timing
randomization is deliberately kept out of the main causal comparison and is
reported as a separate 24/18/12 cm ablation.

## Core and full profiles

Both profiles use three initial states and three policy seeds. Observation,
clutter, and obstacle changes start with three candidate configurations per
task and retain the physically valid subset. Target and receptacle relocation
use only the deterministic 6 and 12 cm tiers.

- **Core:** every retained `(initial state, frozen configuration)` appears
  once, with policy seeds rotated inside each task--change-type cell. This gives
  1,335 matched pairs / 2,670 episodes per model.
- **Full:** crosses every policy seed with every retained configuration. This
  separates policy and environment variation at 4,005 matched pairs / 8,010
  episodes per model.

| Change type | Core pairs | Full pairs |
| --- | ---: | ---: |
| Illumination switch | 360 | 1,080 |
| Camera shift | 360 | 1,080 |
| Target relocation | 186 | 558 |
| Receptacle relocation | 156 | 468 |
| Distractor burst | 99 | 297 |
| Obstacle insertion | 174 | 522 |
| **Total** | **1,335** | **4,005** |

The frozen artifacts are:

```text
benchmark/v1/task_catalog.json
benchmark/v1/core.json
benchmark/v1/full.json
benchmark/v1/physical_preflight.json
benchmark/v1/feasibility_filter.json
benchmark/v1/release_summary.json
benchmark/v1/SHA256SUMS
```

They are versioned `1.0.0`. Every retained physical configuration passed the
static real-MuJoCo preflight. Successful-control trigger coverage and model
outcomes must still be measured during evaluation and are not implied by this
release gate.

## Validity and calibration gates

- Shared setup is identical in control and intervention arms.
- Pre-change action chunks match exactly.
- The proximity trigger fires exactly once and its measured distance is at or
  below the declared threshold.
- Relocated objects remain reachable, supported, collision-free, and within the
  task workspace.
- Inserted distractors and obstacles retain their original object height and do
  not begin in penetration.
- The intervention causes non-zero visual or simulator-state change.
- Missing triggers, collisions introduced at insertion, simulator errors, and
  physically infeasible poses are invalid cases, never policy failures.

## Response-aware extension tracks

Track B adds randomized target updates, receptacle updates, and cancellations.
Track C removes targets or receptacles. They are excluded from the physical
goal-completion leaderboard until model-agnostic scorers can verify task update,
safe stopping, clarification, or infeasibility reporting.

## Required paper reporting

- planned/completed/missing/invalid/duplicate coverage;
- paired outcome table and control-correct regression rate;
- Wilson intervals, paired-bootstrap delta intervals, and exact McNemar tests;
- breakdown by type, family, severity, suite, draw ID, and policy seed;
- pre-change equality, post-event action response, and open-loop exposure;
- Core and Full results reported separately rather than merged;
- safety measurement coverage, not assumed zero violations.
