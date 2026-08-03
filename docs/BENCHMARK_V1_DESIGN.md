# LIBERO-MAX v1 Benchmark Design

Status: deterministic candidate manifests generated; physical calibration and
Cosmos rollout validation are still required before either manifest is frozen.

## Six-type physical main track

The installed catalog contains 40 tasks across LIBERO Spatial, Object, Goal,
and LIBERO-10. Eligibility is defined per intervention rather than forcing an
invalid change into every scene.

| Change type | Eligible tasks | What changes mid-execution |
| --- | ---: | --- |
| Illumination switch | 40 | lights dim sharply or a dim scene lights up |
| Camera shift | 40 | agent-view camera translates and yaws |
| Target relocation | 33 | a target on a floor/table workspace shifts 6 or 12 cm |
| Receptacle relocation | 27 | a movable destination shifts 6 or 12 cm |
| Distractor burst | 12 | five task-native non-target objects appear together |
| Obstacle insertion | 39 | a task-native object appears in the current approach corridor |
| **Task-type cells** | **191** |  |

The target count is 33 rather than the earlier syntactic estimate of 37. One
target begins inside a drawer, and four more begin on a cookie box, ramekin,
stove, or cabinet top. All five are intentionally excluded from free planar
relocation. Receptacle relocation is restricted to movable object receptacles
that start on a floor/table workspace; the benchmark does not translate a
fixed cabinet, stove, or rack merely to inflate coverage.

## Explicit intervention randomness

Policy randomness and intervention randomness are separate fields. A stable
SHA-256-derived intervention seed is computed from:

```text
(sampler version, suite, task, initial-state index, change type, draw id)
```

The generator resolves every sampled value into the manifest. Re-running a
case therefore cannot silently choose a new direction, distractor, or pose.

| Change type | Randomized quantities across draw IDs 0/1/2 |
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
clutter, and obstacle changes use three frozen configurations. Target and
receptacle relocation use only the deterministic 6 and 12 cm tiers.

- **Core:** every unique `(initial state, frozen configuration)` appears once,
  with policy seeds rotated evenly inside each task--change-type cell. This
  gives 1,539 matched pairs / 3,078 episodes per model.
- **Full:** crosses every policy seed with every frozen configuration. This
  separates policy and environment variation at 4,617 matched pairs / 9,234
  episodes per model.

| Change type | Core pairs | Full pairs |
| --- | ---: | ---: |
| Illumination switch | 360 | 1,080 |
| Camera shift | 360 | 1,080 |
| Target relocation | 198 | 594 |
| Receptacle relocation | 162 | 486 |
| Distractor burst | 108 | 324 |
| Obstacle insertion | 351 | 1,053 |
| **Total** | **1,539** | **4,617** |

The generated artifacts on the evaluation host are:

```text
artifacts/libero_task_catalog_v1.json
artifacts/libero_max_v1_core_candidate.json
artifacts/libero_max_v1_full_candidate.json
```

They remain `1.0.0-candidate` until every unique physical configuration passes
preflight and successful-control trigger calibration.

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
