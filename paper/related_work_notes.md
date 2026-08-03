# Related-work audit for LIBERO-MAX

Status: primary-source audit as of 2026-08-02. This is an internal claim ledger,
not a finished bibliography.

## Positioning boundary

The defensible contribution is **not** "the first dynamic VLA benchmark."
Existing work already evaluates moving objects, disturbances, moving obstacles,
and mid-execution intent changes. The sharper LIBERO-MAX contribution is a
matched counterfactual protocol for discrete exogenous world changes introduced
after a policy has committed to execution:

- identical task, initial state, and policy seed across control/change arms;
- exact pre-event action equality as a validity condition;
- semantic event timing rather than only a fixed initial perturbation;
- event-to-next-query exposure for chunked policies;
- paired success decomposition rather than an unpaired intervention success
  rate;
- a common physical track spanning sudden illumination, target relocation, and
  clutter insertion.

## Closest LIBERO extensions

| Work | What it evaluates | Boundary relative to LIBERO-MAX |
| --- | --- | --- |
| [LIBERO](https://arxiv.org/abs/2306.03310) | Knowledge transfer across four manipulation suites | Host task and success predicates; not a change-response protocol |
| [LIBERO-PRO](https://arxiv.org/abs/2510.03827) | Object attributes, **initial** spatial configurations, instructions/tasks, and environments | Perturbed evaluation task configuration, not a matched within-rollout event |
| [LIBERO-Plus](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html) | 10,030 task instances across layout, camera, robot initialization, language, light, texture, and noise | Strongest axis overlap; describes generated task instances and covariate shift, whereas MAX isolates when a change occurs after execution begins |
| [LIBERO-X](https://arxiv.org/abs/2602.06556) | Progressive spatial, topology, visual-attribute, and semantic test levels plus diversified training data | Multi-dimensional OOD difficulty, not paired causal effect of an event |
| [SafeLIBERO](https://arxiv.org/abs/2512.11891) | Static obstacle-interference layouts with task-success and collision-avoidance evaluation | Obstacles exist at initialization; MAX clutter appears suddenly during a matched rollout |

Do not describe LIBERO-Plus, PRO, or X merely as "static benchmarks" without
qualification. Say they perturb the **episode configuration / test task** and
contrast that with MAX's intervention time and paired causal unit.

## Dynamic and interactive evaluations

| Work | Overlap | Distinction to retain |
| --- | --- | --- |
| [DynamicVLA / DOM](https://arxiv.org/abs/2601.22153) | Dynamic objects, speed changes, abrupt motion changes, external disturbances, and latency-aware action streaming | A large-scale benchmark for continuously moving-object manipulation; MAX instead uses discrete event interventions on LIBERO tasks and matched no-change controls |
| [Attention-guided SafeLIBERO moving-obstacle extension](https://arxiv.org/abs/2606.09749) | Non-static obstacles and real-time tracking | A safety-filter evaluation built by extending SafeLIBERO, not the same cross-family matched intervention protocol |
| [ReSteer](https://arxiv.org/abs/2603.17300) | Changes task instructions at intermediate LIBERO states and measures steerability | Directly precedes/overlaps MAX's planned intent track; cite it and do not claim novelty for generic mid-execution instruction changes |
| [Gaze2Act](https://arxiv.org/abs/2605.30282) | Real-robot target-referent change during execution, 30 target-switch trials | Dynamic user-gaze interface and method evaluation, rather than exogenous physical world changes |
| [A3 adaptive execution commitment](https://arxiv.org/abs/2605.11567) | State-dependent action-prefix length and the vulnerability of fixed chunks | Motivates MAX's event-to-query exposure, but does not introduce the same event benchmark |

## Claim language

Safe:

> LIBERO-MAX introduces a matched-pair LIBERO protocol for measuring the causal
> effect of discrete exogenous changes injected after execution begins.

> Unlike episode-level robustness variants, LIBERO-MAX verifies pre-change
> trajectory equality and measures how long a chunked policy continues to
> execute stale actions before it can condition on the changed world.

Avoid until a broader systematic review supports them:

- "the first dynamic VLA benchmark";
- "the first benchmark with mid-execution changes";
- "prior benchmarks only perturb initial states";
- "action divergence proves adaptation";
- any intent-track novelty claim that ignores ReSteer and Gaze2Act.

## Remaining audit items

- Inspect the final code/release protocol of LIBERO-Plus to confirm that no
  official evaluation mode changes perturbations during an active rollout.
- Inspect DynamicVLA's released DOM manifests for whether paired no-change
  controls or event timestamps are standardized in code.
- Add method/checkpoint citations for every evaluated model only after the
  roster is frozen.
- Track later work that combines discrete physical events with matched
  counterfactual rollouts; narrow the novelty wording if one is found.
