# LIBERO-MAX Hard: dynamic evaluation on LIBERO-Plus

Status: candidate construction and simulator preflight

## Purpose

LIBERO-MAX Hard composes the static distribution shifts in LIBERO-Plus with a
second, exogenous change that occurs after the policy has started acting. It is
therefore not a replacement name for LIBERO-Plus and not a collection of
easier original-LIBERO tasks.

The required comparison ladder is:

1. original LIBERO;
2. LIBERO-Plus static shift;
3. MAX-Hard: the same Plus task plus one mid-execution event;
4. MAX-Compose: multiple ordered events, reserved for the composition study.

MAX-Hard is considered empirically harder only after policy rollouts show a
paired decrease relative to the Plus substrate. Dataset size and visual change
alone are not evidence for that claim.

## Frozen candidate design

| Profile | Selection | Matched pairs | Rollouts |
| --- | --- | ---: | ---: |
| Core | 7 Plus categories x 5 difficulty levels x 40 tasks | 1,400 | 2,800 |
| Full | every task in the installed Plus benchmark | 10,030 | 20,060 |

Core contains exactly 200 pairs per Plus category, 280 per Plus difficulty
level, and 175 per dynamic event. Full retains all 10,030 tasks and assigns the
eight events as evenly as eligibility permits (1,253 or 1,254 pairs each).
Core is an exact subset of Full.

The paired arms fix suite, task index, Plus task variant, initial-state index,
policy seed, resolved event parameters, and event seed. The intervention arm
replays the control action chunks exactly until the event fires.

## Dynamic events

All main-track events fire when the end effector first comes within 18 cm of
the task's primary target. The threshold is state-based, not wall-clock-based.
The event parameters are materialized in the manifest; rollout code does not
sample hidden randomness.

| Event | Resolved variation | Physical task remains possible |
| --- | --- | --- |
| Illumination switch | 0.20x dimming or 1.80x brightening | geometry unchanged |
| Camera shift | fixed translation, yaw, and field-of-view change | geometry unchanged |
| Visual theme switch | one of two fixed RGB material transforms | geometry unchanged |
| Sensor-noise onset | seeded noise plus 8% or 16% occlusion | simulator state unchanged |
| Target relocation | fixed direction, 6 cm or 12 cm | same planar support |
| Receptacle relocation | fixed direction, 6 cm or 12 cm | same planar support |
| Distractor burst | 5 or up to 8 seeded distractors | target and goal retained |
| Obstacle insertion | same-support object on a fixed target-approach ring | alternate path retained |

The two relocation directions are deterministic per task, never randomly
resampled per rollout. Candidate directions must pass real-MuJoCo stability,
support, contact, and visibility checks before release. Failed task/event
configurations are replaced by another eligible task in the same Core stratum;
Full keeps the task and falls back to a non-geometric observation event. This
can introduce a small Full event-count imbalance after feasibility filtering;
Core remains exactly balanced. No infeasible case is counted as a policy
failure.

## LIBERO-Plus compatibility

The installed Plus benchmark exposes 10,030 tasks across `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`. Of these, 6,287 use virtual
filenames encoding camera and robot-initial-state parameters. The catalog
builder mirrors the Plus environment wrapper: it resolves the virtual suffix
to the underlying BDDL while preserving every encoded parameter and validates
the task name against `task_classification.json`.

The audited raw catalog contains:

- 10,030/10,030 named and classified tasks;
- 8,236 tasks with a planar movable target;
- 6,632 tasks with a planar movable receptacle;
- 3,391 tasks with at least five catalog distractors.

## Release gates

A candidate can be renamed from `2.0.0-candidate` to a release version only if:

1. catalog coverage is exactly 10,030/10,030;
2. Core has exactly 1,400 pairs and its 7 x 5 x 40 design is exact;
3. Full has exactly 10,030 pairs and contains Core exactly;
4. every resolved Core and Full transition passes real-MuJoCo preflight;
5. every intervention rollout records exactly one reached event;
6. control/intervention initial-state hashes and pre-event action chunks match;
7. policy results report complete keyed coverage, paired bootstrap intervals,
   exact McNemar tests, Plus category, Plus difficulty, event type, and draw;
8. the observed MAX-Hard drop is separated from failures already present on
   the static Plus task.

The preflight gate establishes simulator validity, not empirical difficulty.
The latter requires complete paired policy rollouts.
