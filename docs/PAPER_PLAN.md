# LIBERO-MAX Paper Plan

Status: executable Track A pilot; Track B/C scorers remain implementation work.

The first paper release centers on three intuitive, visually demonstrable
stressors: sudden light off/on, sudden target relocation, and the sudden
appearance of multiple distractor objects.

## Central question

Do VLA policies appropriately revise their behavior when an external change is
introduced after execution begins, and which kinds of changes expose the
largest gap between static task competence and mid-execution responsiveness?

## Benchmark tracks

| Track | Families | Correctness signal | Current status |
| --- | --- | --- | --- |
| A. Physical adaptation | `OBS`, `GEO`, `CLUTTER` | original LIBERO goal completion, paired against no-change control | executable for Cosmos |
| B. Intent revision | `INTENT` | alternate goal predicate, safe cancellation, or clarification response | runtime core only; Cosmos language-query hook and scorers missing |
| C. Feasibility awareness | `FEAS` | safe abstention plus explicit infeasibility report | taxonomy only; action/text protocol and scorer missing |

Track A is the first empirical paper milestone. Track B and C must not reuse
ordinary LIBERO task success as their correctness label.

## Proposed contributions

1. A matched, deterministic protocol that isolates the effect of a change from
   base-task competence and rollout randomness.
2. A taxonomy spanning observation, geometry, obstruction, intent, and
   feasibility changes with response-mode-aware scoring.
3. A diagnostic decomposition into preserved success, recovery, regression,
   and persistent failure, with coverage and category-level uncertainty.
4. A cross-model study of frozen VLA policies, observation-history variants,
   explicit replanning, online adaptation, and an oracle change-aware reference.

## Experimental design

- Use exact `(suite, task, initial-state, policy-seed)` matching across arms.
- Calibrate low/medium/high severities so each change is visible and physically
  valid without causing simulator instability.
- Place interventions at semantic triggers and explicitly measure any delay to
  the next policy query; later ablations cover early, middle, and late timing.
- Use target-proximity triggers for the core pilot and report both the physical
  change step and the next policy-query step; later timing ablations vary the
  distance threshold.
- Report full manifest coverage, invalid cases, paired flips, Wilson intervals,
  paired-bootstrap intervals, and exact McNemar tests.
- Separate change detection, response latency, final correctness, and safety.
- Verify exact pre-change action-chunk equality and report the first post-event
  action-chunk difference. This is behavioral-response evidence, not by itself
  proof of successful adaptation.

## Main tables

1. Overall paired outcome by model and benchmark track.
2. Breakdown by change family, severity, and timing.
3. Control-correct subset regression and recovery analysis.
4. Response latency and safety measurement coverage.
5. Ablations: observation history, closed-loop query interval, explicit change
   notification, and online adaptation.

## Claim gates

- No aggregate model ranking before every planned pair is complete or declared
  invalid under a pre-specified rule.
- No adaptation claim from goal preservation alone; show behavior or action
  change conditioned on the intervention and report latency coverage.
- No safety claim when collision/stop instrumentation coverage is incomplete.
- No intent or infeasibility score from the original LIBERO `done` predicate.

## Immediate empirical ladder

1. Five-case Cosmos physical pilot on one task and one initial state.
2. Severity calibration across three initial states and three policy seeds.
3. Expand to representative tasks from Spatial, Object, Goal, and LIBERO-10.
4. Add at least two model families and the oracle-notified diagnostic.
5. Freeze the v1 manifest before the main evaluation.

The current task-0 mechanics and 18 cm trigger calibration are recorded in
[`PILOT_CALIBRATION.md`](PILOT_CALIBRATION.md).
