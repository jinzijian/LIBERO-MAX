# LIBERO-MAX v1 Benchmark Design

Status: proposed scale-up after the task-0 Cosmos pilot; the v1 manifest is not
yet frozen.

## Empirical task universe

The catalog builder inspected the installed BDDL definitions for all 40 tasks
in LIBERO Spatial, Object, Goal, and LIBERO-10:

| Suite | Tasks | Manipulated target identified | At least 5 native distractors |
| --- | ---: | ---: | ---: |
| Spatial | 10 | 10 | 0 |
| Object | 10 | 10 | 10 |
| Goal | 10 | 8 | 0 |
| LIBERO-10 | 10 | 10 | 2 |
| **Total** | **40** | **38** | **12** |

Lighting can cover all 40 tasks. Target relocation is restricted to the 38
tasks with an identified manipulated object. Five-object distractor bursts are
restricted to the 12 tasks with enough task-native non-target objects; v1 does
not silently inject unseen asset classes into the other scenes.

## Proposed main matrix

Every case uses three initial states and three deterministic policy seeds.

| Axis | Task coverage | Conditions | Matched pairs |
| --- | ---: | ---: | ---: |
| Illumination switch | 40 | light off, light on | 720 |
| Target relocation | 38 | 6 cm, 12 cm | 684 |
| Distractor burst | 12 | 3 objects, 5 objects | 216 |
| **Total per model** |  |  | **1,620** |

Each matched pair contains a no-change control and an intervention episode, so
the matrix requires 3,240 episodes per model before any invalid-case reruns.

## Trigger protocol

The main matrix uses an 18 cm end-effector-to-target trigger. The intervention
fires immediately on the first threshold crossing, even inside an open-loop
action chunk. Reports include the physical change step, next policy-query step,
and intervening open-loop exposure.

A timing ablation on a pre-registered representative task subset compares 24,
18, and 12 cm. Trigger coverage must be reported: episodes that never reach the
threshold are invalid intervention cases, not ordinary policy failures.

## Severity and validity

- Lighting-off uses 5% of the current diffuse/specular intensity.
- Lighting-on starts both arms at 5%, then restores the intervention arm with a
  calibrated x20 change.
- Target displacement is applied in a task-valid surface direction; the 6 cm
  and 12 cm destinations must remain collision-free and reachable.
- Distractors are hidden off-world in both arms and restored only in the
  intervention arm at prevalidated poses.
- A case is valid only if the setup is identical across arms, the trigger fires,
  the simulator remains stable, and the intervention produces a non-zero visual
  or state change.

## Baseline roster

1. Frozen VLA, standard 16-action chunks.
2. Frozen VLA with shorter closed-loop query intervals.
3. Frozen VLA with explicit change notification (oracle diagnostic).
4. Observation-history / replanning baseline.
5. Online-adaptation method, if its update contract is fully specified.

The oracle-notified arm measures headroom from knowing that a change occurred;
it is not a deployable benchmark winner.

## Required paper reporting

- full planned/completed/missing/invalid/duplicate coverage;
- preserved successes, gains, regressions, and persistent failures;
- Wilson intervals, paired-bootstrap delta intervals, and exact McNemar tests;
- breakdown by axis, severity, trigger threshold, task suite, and base-control
  correctness;
- pre-change action equality, first post-change action difference, open-loop
  exposure, and final goal outcome;
- measured safety coverage rather than assumed zero violations.
