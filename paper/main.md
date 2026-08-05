# LIBERO-MAX: Do Vision-Language-Action Models Adapt When the World Changes Mid-Execution?

## Abstract

Vision-language-action models are commonly evaluated in episodes whose visual
conditions, object locations, and distractor sets remain fixed after execution
begins. Real robot environments do not satisfy this assumption. We introduce
LIBERO-MAX, a matched-pair benchmark for **mid-execution adaptation to
exogenous changes**. The v1 physical track covers six controlled changes:
illumination switches, camera shifts, target and receptacle relocation,
five-object distractor bursts, and path-obstacle insertion. Each intervention
fires when the robot approaches the task target and is paired with a no-change
rollout sharing the task, initial state, policy seed, and deterministic setup.
Independent intervention draws vary severity, identity, and placement while
using one prevalidated relocation direction per task and resolving every value
into an immutable manifest. The
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

The v1 physical-completion track contains six changes:

1. **Illumination switch:** the lights suddenly turn off or on.
2. **Camera shift:** the external camera suddenly translates and rotates.
3. **Target relocation:** the object being approached moves to a new reachable
   pose.
4. **Receptacle relocation:** the destination moves while the original goal
   remains feasible.
5. **Distractor burst:** five task-native non-target objects suddenly
   appear.
6. **Obstacle insertion:** a task-native object appears along the current
   approach corridor.

These axes cover perceptual degradation, viewpoint shift, geometric
replanning, target identity under clutter, and collision-aware detours. A
separate 96-pair Intent Core changes the target or receptacle named by the
instruction, or cancels the task. It uses alternate BDDL goal predicates and a
frozen ten-step safe-stop contract rather than silently reusing the original
LIBERO success predicate. Explicit infeasibility awareness remains outside v1
because it requires a model-agnostic abstention interface.

Our contributions are:

- a deterministic matched-pair protocol for evaluating changes introduced
  after execution begins;
- six simulator-grounded intervention types with semantic proximity triggers,
  deterministic random draws, severity controls, and explicit validity checks;
- a response-aware intent-revision track covering target updates, receptacle
  updates, and task cancellation;
- reporting that decomposes paired outcomes and includes coverage, uncertainty,
  pre-change action equality, post-change action response, and open-loop
  exposure;
- [AFTER EXPERIMENTS] a cross-model analysis identifying which changes and
  execution regimes produce the largest adaptation gaps.

## 2. Related Work

### Vision-language-action evaluation

[LIBERO](https://arxiv.org/abs/2306.03310) established four suites for
knowledge transfer in language-conditioned manipulation. Recent extensions
broaden the evaluation distribution. LIBERO-PRO varies object attributes,
initial spatial configurations, instructions, tasks, and environments
([Jiang et al., 2025](https://arxiv.org/abs/2510.03827)); LIBERO-Plus generates
progressive variants over object layout, camera pose, robot initial state,
language, illumination, texture, and sensor noise
([Fei et al., 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html));
and LIBERO-X jointly increases spatial, object, scene-topology, and semantic
difficulty ([Zhang et al., 2026](https://arxiv.org/abs/2602.06556)). These
benchmarks diagnose generalization to a perturbed episode configuration. Our
question is complementary: after a matched pair has followed the same policy
trajectory, can the policy respond to a change injected into one arm during
execution?

### Robust and dynamic manipulation benchmarks

Static obstacle benchmarks add an important safety dimension. SafeLIBERO uses
pre-placed obstacle-interference scenarios with randomized episode layouts
([Hu et al., 2025](https://arxiv.org/abs/2512.11891)), whereas a later
SafeLIBERO extension studies moving obstacles for real-time safety filtering
([Park et al., 2026](https://arxiv.org/abs/2606.09749)). DynamicVLA's Dynamic
Object Manipulation benchmark instead centers on continuously moving objects,
including speed variation, abrupt motion changes, and disturbances
([Xie et al., 2026](https://arxiv.org/abs/2601.22153)). LIBERO-MAX does not
claim to be the first dynamic manipulation evaluation. Its unit of analysis is
different: deterministic control/intervention pairs for discrete exogenous
events, with pre-event action equality and event-to-query exposure measured
explicitly across lighting, geometry, and clutter.

Mid-execution intent change has also been studied directly. ReSteer switches
LIBERO-Goal instructions at sampled intermediate states and measures
state-dependent steerability
([Liu et al., 2026](https://arxiv.org/abs/2603.17300)); Gaze2Act evaluates a
real-robot target switch conveyed by human gaze
([Zuo et al., 2026](https://arxiv.org/abs/2605.30282)). These results establish
that instruction or referent revision is a real and nontrivial problem. For
that reason, LIBERO-MAX evaluates intent revision in a separate response-aware
track rather than folding it into the physical-completion score.

### Online adaptation and replanning

Action chunking creates a perception--execution gap because the world may
change while previously predicted actions are still being applied. Adaptive
execution-commitment methods vary the accepted action prefix based on
state-dependent reliability ([Chen et al., 2026](https://arxiv.org/abs/2605.11567)).
LIBERO-MAX exposes this issue through an evaluator-side quantity: the number of
environment steps between the physical event and the first policy query that
can observe it. We compare shorter query intervals, observation history,
explicit change notification, and online adaptation, but do not equate a
parameter update or a changed action chunk with correct episode-level
adaptation; the changed arm must still satisfy its response-aware outcome.

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

| Type | Randomized control | Expected behavior |
| --- | --- | --- |
| Illumination switch | light-off scale or dim-to-on setup | preserve target perception and continue |
| Camera shift | fixed global direction, translation magnitude, and yaw | recover viewpoint correspondence and continue |
| Target relocation | one calibrated task direction at deterministic 6 or 12 cm tiers | relocalize the target and revise the reach |
| Receptacle relocation | one calibrated task direction at deterministic 6 or 12 cm tiers | revise the placement plan |
| Distractor burst | five task-native identities/placements | preserve target identity and continue safely |
| Obstacle insertion | identity, approach-path fraction, and lateral offset | detour without collision and complete the task |

All inserted distractors and obstacles are task-native objects. They are hidden
off-world in both paired arms before execution and restored only in the
intervention arm. Each case records a sampler version, draw ID, intervention
seed, and fully resolved parameters; there is no hidden runtime sampling.

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

### 3.5 Intent revision

Intent Core contains 96 deterministic matched pairs: 30 target updates, 18
receptacle updates, and 48 cancellations. Updates are correct only when the
alternate task predicate is satisfied after the model has received the new
instruction. Cancellation is correct only when, during the first ten executed
steps after the updated instruction reaches the policy, end-effector net
motion is at most 2 cm, path length is at most 4 cm, target motion is at most
1 cm, and the superseded goal is not completed. All alternate predicates and
trigger entities are instantiated in the source environment and are checked
to be unsatisfied before rollout.

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
Goal, and LIBERO-10. After deterministic task--change--draw pruning, the frozen
release contains 177 task--change-type cells. Illumination and camera changes
apply to all 40 tasks; retained planar target relocation applies to 31,
movable-receptacle relocation to 26, five-object bursts to 12, and task-native
obstacle insertion to 28.

The v1 candidate provides two profiles:

| Type | Eligible tasks | Core pairs | Full pairs |
| --- | ---: | ---: | ---: |
| Illumination switch | 40 | 360 | 1,080 |
| Camera shift | 40 | 360 | 1,080 |
| Target relocation | 31 | 186 | 558 |
| Receptacle relocation | 26 | 156 | 468 |
| Distractor burst | 12 | 99 | 297 |
| Obstacle insertion | 28 | 174 | 522 |
| **Total** |  | **1,335** | **4,005** |

Core evaluates every unique frozen configuration once and rotates three policy
seeds evenly within each task--change-type cell. Full crosses each policy seed
with every configuration to separate policy and intervention variation. The v1
manifest is frozen only after task-specific relocation and insertion poses pass
physical preflight and trigger-coverage calibration.

Intent Core adds 96 matched pairs and uses the same 18 cm target-proximity
trigger. Its correctness label is response-aware and is therefore reported
separately from physical goal completion.

## 6. Experimental Setup

### Models and baselines

1. Cosmos Policy Predict2-2B with its default 16-step action commitment.
2. pi0.5-LIBERO with its official five-step replanning interval and
   deterministic matched flow noise.
3. Cosmos Policy on a frozen 180-pair, six-axis-balanced subset with a
   five-step closed-loop query interval.
4. Cosmos Policy on the same subset with an evaluator-provided event
   notification appended to the task instruction.

The event-notified diagnostic measures headroom from knowing that a change
occurred and is not a deployable baseline.

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

| Model | Light | Camera | Target move | Receptacle move | Burst-5 | Obstacle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [pending] |  |  |  |  |  |  |

### 7.3 Timing and closed-loop exposure

[Five-step query-interval and event-notification ablations on the frozen
180-pair subset; trigger coverage, open-loop exposure, and post-change action
response.]

### 7.4 Failure analysis

[Separate failure to detect, stale open-loop execution, target confusion,
failed replanning, simulator-invalid cases, and base-task failures.]

## 8. Limitations

The first release uses simulation, task-native inserted objects, and a geometric
proximity trigger available to the evaluator but not exposed to the policy.
Goal completion alone cannot establish change detection. Physical-track safety
claims remain limited because robot--object collision instrumentation is not
complete. Intent scoring covers instruction updates and cancellation, but
explicit infeasibility reporting remains future work until a model-agnostic
abstention interface is available.

## 9. Conclusion

[Write only after the frozen evaluation is complete.]
