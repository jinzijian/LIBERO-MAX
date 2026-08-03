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
| Target relocation | 37 | the approached movable target shifts 6 or 12 cm |
| Receptacle relocation | 27 | a movable destination shifts 6 or 12 cm |
| Distractor burst | 12 | five task-native non-target objects appear together |
| Obstacle insertion | 39 | a task-native object appears in the current approach corridor |
| **Task-type cells** | **195** |  |

The target count is 37 rather than the earlier catalog estimate of 38 because
one target begins inside a drawer. It is intentionally excluded from planar
relocation. Receptacle relocation is restricted to movable object receptacles;
the benchmark does not translate a fixed cabinet, stove, or rack merely to
inflate coverage.

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
| Illumination | mild dim scale in [0.35, 0.55], near-dark scale in [0.02, 0.08], or dim-to-on setup in [0.03, 0.10] |
| Camera | random planar direction at 2/4/6 cm and random yaw sign at 5/10/15 degrees |
| Target relocation | random planar direction; displacement remains exactly 6 or 12 cm |
| Receptacle relocation | random planar direction; displacement remains exactly 6 or 12 cm |
| Distractor burst | task-native identity subset/order and five randomized non-overlapping angular placements around the target |
| Obstacle insertion | task-native obstacle identity, path fraction in [0.35, 0.65], and signed lateral offset |

The semantic trigger remains fixed at 18 cm in the main table. Timing
randomization is deliberately kept out of the main causal comparison and is
reported as a separate 24/18/12 cm ablation.

## Core and full profiles

Both profiles use three initial states, three policy seeds, and intervention
draw IDs 0/1/2.

- **Core:** a balanced Latin-square assignment couples one draw to each
  `(initial state, policy seed)` cell. Every policy seed sees every draw once
  across the three initial states. This gives 1,755 matched pairs / 3,510
  episodes per model.
- **Full:** crosses every policy seed with all three intervention draws. This
  separates policy and environment variation at 5,265 matched pairs / 10,530
  episodes per model.

| Change type | Core pairs | Full pairs |
| --- | ---: | ---: |
| Illumination switch | 360 | 1,080 |
| Camera shift | 360 | 1,080 |
| Target relocation | 333 | 999 |
| Receptacle relocation | 243 | 729 |
| Distractor burst | 108 | 324 |
| Obstacle insertion | 351 | 1,053 |
| **Total** | **1,755** | **5,265** |

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
