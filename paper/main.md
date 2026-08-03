# LIBERO-MAX: Do Vision-Language-Action Models Adapt When the World Changes Mid-Execution?

## Abstract

Vision-language-action models are commonly evaluated in episodes whose visual
conditions, object locations, and distractor sets remain fixed after execution
begins. Real robot environments do not satisfy this assumption. We introduce
LIBERO-MAX, a matched-pair benchmark for **mid-execution adaptation to
exogenous changes**. The first release centers on three controlled changes that
are easy to interpret and reproduce: sudden illumination switches, target
relocation, and the sudden appearance of multiple distractor objects. Each
intervention fires when the robot approaches the task target, and is paired
with a no-change rollout sharing the task, initial state, and policy seed. The
protocol separates preserved capability, intervention-side recovery,
change-induced regression, and persistent failure, while recording the delay
between physical change and the next policy query. We evaluate [MODEL ROSTER]
on [FROZEN V1 MANIFEST SIZE] matched pairs. [RESULT SENTENCES ONLY AFTER FULL
COVERAGE.] LIBERO-MAX exposes the gap between static task competence and the
ability to respond when the world changes during execution.

## 1. Introduction

A robot can appear competent when the world remains exactly as it was at the
start of an episode. That competence is incomplete. During a real task, a room
light can be switched off, a person can move the object the robot is reaching
for, or previously absent objects can enter the scene. A useful policy must not
only solve the original task; it must notice that its assumptions are stale
and condition subsequent behavior on the new world state.

Existing manipulation evaluations largely measure success under a fixed
episode configuration or perturb the initial state before execution [CITES].
Those settings cannot answer a distinct question: **what happens when a policy
has already committed to a trajectory and an external change invalidates its
current perception or plan?** An ordinary intervention success rate is also
insufficient. A failure may simply reflect that the policy could not solve the
base task, while a success may occur without any change-conditioned response.

LIBERO-MAX addresses these issues with deterministic matched pairs. A control
episode and an intervention episode share the task, initial-state index, and
policy seed. Their actions are required to match before the change. The change
then fires at a semantic target-proximity trigger rather than an arbitrary wall
clock time. This design isolates change-induced regressions from base-task
failures and exposes the open-loop delay before the policy can observe and
react to the new state.

The first benchmark release deliberately focuses on three physical changes:

1. **Illumination switch:** the lights suddenly turn off or on.
2. **Target relocation:** the object being approached moves to a new reachable
   pose.
3. **Distractor burst:** multiple task-native non-target objects suddenly
   appear.

These axes cover perceptual degradation, geometric replanning, and target
identity under clutter without requiring a model-specific textual refusal or
termination interface. Intent revision and infeasibility awareness are
important extensions, but they require response-aware scorers and are not
silently reduced to the original LIBERO success predicate.

Our contributions are:

- a deterministic matched-pair protocol for evaluating changes introduced
  after execution begins;
- three simulator-grounded intervention axes with semantic proximity triggers,
  severity controls, and explicit validity checks;
- reporting that decomposes paired outcomes and includes coverage, uncertainty,
  pre-change action equality, post-change action response, and open-loop
  exposure;
- [AFTER EXPERIMENTS] a cross-model analysis identifying which changes and
  execution regimes produce the largest adaptation gaps.

## 2. Related Work

### Vision-language-action evaluation

[LIBERO, OpenVLA, Octo, RT-X, Cosmos Policy citations and exact evaluation
assumptions.]

### Robust and dynamic manipulation benchmarks

[Initial-state perturbations, visual robustness, dynamic scenes, disturbance
recovery, and interactive instruction benchmarks. Distinguish changes before
execution from changes during execution.]

### Online adaptation and replanning

[Test-time adaptation, observation history, closed-loop replanning, world
models, and explicit change detection. Do not equate parameter updates with
successful episode-level adaptation.]

## 3. LIBERO-MAX

### 3.1 Evaluation unit

The evaluation unit is a matched pair

\[
(\tau_i^{\mathrm{control}}, \tau_i^{\mathrm{change}}),
\]

with an exact shared key `(suite, task index, initial-state index, policy seed)`.
The control receives the scenario setup but no mid-execution change. The
intervention receives the same setup and exactly one change.

### 3.2 Semantic trigger

Let \(p_t^{ee}\) be the end-effector position and \(p_t^{target}\) the target
object position. The core trigger fires at the first step

\[
t^* = \min\{t : \|p_t^{ee} - p_t^{target}\|_2 \le d\}.
\]

The pilot uses \(d=0.18\) m. The physical intervention occurs immediately,
including inside an open-loop action chunk. If the next policy query is at
\(q(t^*)\), open-loop exposure is \(q(t^*)-t^*\).

### 3.3 Intervention axes

| Axis | Low / medium / high control | Expected behavior |
| --- | --- | --- |
| Illumination | calibrated light intensity or direction | preserve target perception and continue |
| Target relocation | task-valid in-plane displacement | relocalize the target and revise the reach |
| Distractor burst | number and similarity of inserted objects | preserve target identity and continue safely |

All inserted distractors are task-native objects. They are hidden off-world in
both paired arms before execution and restored only in the intervention arm.

### 3.4 Validity contract

A matched pair is valid only when:

- both arms load the declared task and initial-state index;
- the serialized initial state and pre-change action chunks match;
- shared setup operations succeed in both arms;
- the semantic trigger fires exactly once;
- the intervention produces a non-zero visual or simulator-state change;
- neither rollout ends in an infrastructure or simulator error.

Unreached triggers and invalid interventions are reported separately; they are
not counted as ordinary policy failures.

## 4. Metrics

For pair \(i\), let \(c_i\) and \(z_i\) be control and intervention
correctness. We report the complete 2x2 outcome table:

| Control | Intervention | Outcome |
| --- | --- | --- |
| 1 | 1 | preserved capability |
| 0 | 1 | intervention-side recovery |
| 1 | 0 | change-induced regression |
| 0 | 0 | persistent failure |

Primary metrics are intervention accuracy, paired robustness delta
\(\frac{1}{N}\sum_i(z_i-c_i)\), and regression rate conditioned on control
success. Wilson intervals accompany accuracy, paired bootstrap intervals
accompany deltas, and exact McNemar tests use the two discordant cells.

Diagnostic metrics include trigger coverage, open-loop exposure, the first
post-change action-chunk difference, episode length, and measured safety
coverage. Action difference is evidence that policy behavior changed; it is
not sufficient evidence that adaptation was correct.

## 5. Benchmark Construction

The installed task catalog contains 40 tasks across LIBERO Spatial, Object,
Goal, and LIBERO-10. Lighting applies to all 40. A manipulated target is
identified in 38 tasks. Twelve tasks have at least five task-native distractors.

The proposed v1 matrix contains 1,620 matched pairs per model:

| Axis | Tasks | Conditions | Initial states | Policy seeds | Pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Illumination | 40 | 2 | 3 | 3 | 720 |
| Target relocation | 38 | 2 | 3 | 3 | 684 |
| Distractor burst | 12 | 2 | 3 | 3 | 216 |
| **Total** |  |  |  |  | **1,620** |

The v1 manifest is frozen only after task-specific relocation and distractor
poses pass physical preflight and trigger-coverage calibration.

## 6. Experimental Setup

### Models and baselines

1. Frozen VLA with its default action chunk.
2. The same VLA with shorter closed-loop query intervals.
3. Observation-history or explicit replanning baseline.
4. Online-adaptation method with a fully specified update contract.
5. Oracle change-notified diagnostic.

The oracle measures headroom from knowing that a change occurred and is not a
deployable baseline.

### Reproducibility

Report checkpoint hashes, text-embedding assets, normalization statistics,
action chunk, query interval, simulator versions, manifest hash, GPU type, and
all missing/invalid/duplicate cases.

## 7. Results

### 7.1 Main paired results

| Model | Control acc. | Change acc. | Delta | Regressions | Recoveries | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [pending] |  |  |  |  |  |  |

### 7.2 By intervention axis

| Model | Light off | Light on | Move 6 cm | Move 12 cm | Burst 3 | Burst 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [pending] |  |  |  |  |  |  |

### 7.3 Timing and closed-loop exposure

[24/18/12 cm trigger ablation; query-interval ablation; trigger coverage and
post-change action response.]

### 7.4 Failure analysis

[Separate failure to detect, stale open-loop execution, target confusion,
failed replanning, simulator-invalid cases, and base-task failures.]

## 8. Limitations

The first release uses simulation, task-native distractors, and a geometric
proximity trigger available to the evaluator but not exposed to the policy.
Goal completion alone cannot establish change detection. Safety claims require
instrumented collision or termination coverage. Intent revision and explicit
infeasibility reporting remain separate tracks until model-agnostic response
interfaces and scorers are implemented.

## 9. Conclusion

[Write only after the frozen evaluation is complete.]
