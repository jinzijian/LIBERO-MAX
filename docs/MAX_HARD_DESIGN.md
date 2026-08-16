# LIBERO-MAX-5600: dynamic evaluation on LIBERO-Plus

Status: released as benchmark version 2.0.0

## Purpose

LIBERO-MAX-5600 composes the static distribution shifts in LIBERO-Plus with a
second, exogenous change that occurs after the policy has started acting. It is
therefore not a replacement name for LIBERO-Plus and not a collection of
easier original-LIBERO tasks.

The required comparison ladder is:

1. original LIBERO;
2. LIBERO-Plus static shift;
3. LIBERO-MAX-5600: the same Plus task plus one mid-execution event;
4. MAX-Compose: multiple ordered events, reserved for the composition study.

LIBERO-MAX is considered empirically harder only after policy rollouts show a
paired decrease relative to the Plus substrate. Dataset size and visual change
alone are not evidence for that claim.

## Frozen release design

| Track | Selection | Matched pairs | Rollouts per model |
| --- | --- | ---: | ---: |
| Physical | 7 Plus categories x 800 tasks | 5,600 | 11,200 |
| Intent | instruction update and cancellation | 96 | 192 |

The physical track contains exactly 800 pairs per Plus category and 700 per
dynamic event. Each event has two deterministic draws with 350 pairs each;
every category/event/draw cell therefore contains exactly 50 pairs. The five
Plus difficulty levels are balanced as closely as source capacity permits.
The former 1,400-case Core is only a development seed: 1,399 cases are retained
and one case that failed the final five-object clutter contract is replaced.

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
| Distractor burst | exactly 5 seeded distractors in both draws | target and goal retained |
| Obstacle insertion | same-support object on a fixed target-approach ring | alternate path retained |

The two relocation directions are deterministic per task, never randomly
resampled per rollout. Candidate directions must pass real-MuJoCo stability,
support, contact, and visibility checks before release. Failed task/event
configurations are replaced by another eligible task while preserving the
exact category/event/draw quotas. This rule also applies to development-Core
cases. No infeasible case is counted as a policy failure.

Obstacle failures are conservatively propagated across Plus variants with the
same physical scene signature so camera, lighting, language, or robot-start
variants do not repeatedly select the same bad placement. Selection is solved
as a deterministic capacity-constrained matching, so every released balance
cell remains exact after replacements.

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
2. the physical manifest has exactly 5,600 unique Plus tasks;
3. all 7 category, 8 event, 2 draw, and 112 joint-cell quotas are exact;
4. every resolved transition passes real-MuJoCo preflight;
5. every intervention rollout records exactly one reached event;
6. control/intervention initial-state hashes and pre-event action chunks match;
7. policy results report complete keyed coverage, paired bootstrap intervals,
   exact McNemar tests, Plus category, Plus difficulty, event type, and draw;
8. the observed MAX drop is separated from failures already present on
   the static Plus task.

The preflight gate establishes simulator validity, not empirical difficulty.
The latter requires complete paired policy rollouts.
