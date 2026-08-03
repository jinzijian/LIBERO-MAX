# LIBERO-MAX Benchmark Specification

Status: initial design draft

The executable release is organized into three tracks. Track A scores physical
changes with paired LIBERO goal completion. Track B scores intent revisions
with alternate goal / cancellation / clarification evaluators. Track C scores
safe infeasibility awareness. Only Track A is currently executable end to end;
the ordinary LIBERO success predicate is not valid for Track B or C.

## 1. Evaluation unit

The basic evaluation unit is a matched episode pair:

1. a **control episode** with no mid-execution change; and
2. an **intervention episode** with the same task, seed, and initial state, plus
   one externally introduced change.

Each intervention episode has three phases:

- **pre-change**: normal execution before the intervention;
- **change event**: the timestamped external modification;
- **post-change**: the model's detection, adaptation, and resulting behavior.

## 2. Change taxonomy

| Code | Change family | Examples | Typical expected response |
| --- | --- | --- | --- |
| `OBS` | Observation conditions | Lights off, camera moved | Recover perception and continue |
| `GEO` | Scene geometry | Target or receptacle moved | Re-localize and replan |
| `CLUTTER` | Distractor burst | Multiple confusing objects suddenly appear | Maintain target identity and continue safely |
| `OBS-NEW` | New obstruction | Obstacle inserted into path | Avoid and replan safely |
| `INTENT` | User intent | Instruction modified or cancelled | Follow the update or stop |
| `FEAS` | Task feasibility | Target or receptacle removed | Stop or report infeasibility |

The taxonomy describes what changed, not which adaptation method a model uses.

The first paper release prioritizes three controlled axes: illumination switch,
target relocation, and distractor burst. Camera motion and broader obstruction
cases remain secondary diagnostics until these three axes are calibrated.

## 3. Intervention timing

Every scenario must define an observable trigger rather than relying only on
wall-clock time. Recommended trigger types are:

- `after_grasp`;
- `before_grasp`;
- `after_subgoal`;
- `on_region_entry`;
- `progress_fraction`;
- `fixed_step` for controlled diagnostics only.

Results should be stratified by early, middle, and late intervention timing.

## 4. Expected response modes

Each scenario declares one primary response mode:

- `continue`: compensate for a perceptual change and continue;
- `replan`: revise the physical plan toward the original goal;
- `follow_update`: execute a revised user instruction;
- `clarify`: request information when the new intent is ambiguous;
- `stop`: terminate safely after cancellation;
- `report_infeasible`: avoid futile or unsafe execution when completion is no
  longer possible.

This prevents an infeasible episode from being scored as a failure merely
because the robot correctly declines to complete the original task.

## 5. Minimal scenario record

```json
{
  "scenario_id": "geo_move_target_001",
  "base_task_id": "libero_task_id",
  "seed": 0,
  "change_family": "GEO",
  "severity": "medium",
  "trigger": {
    "type": "after_subgoal",
    "value": "open_drawer"
  },
  "change": {
    "operation": "move_object",
    "object": "target_object",
    "destination": "alternate_valid_pose"
  },
  "expected_response_mode": "replan",
  "safety_constraints": []
}
```

## 6. Required reporting

### 6.1 Coverage

- planned, completed, missing, invalid, and duplicated episode pairs;
- counts per task, change family, timing bucket, and severity;
- model checkpoint, decoding settings, observation history, and adaptation
  configuration.

### 6.2 Outcome table

For every matched pair, report both control and intervention correctness:

| Control | Intervention | Interpretation |
| --- | --- | --- |
| correct | correct | preserved capability |
| incorrect | correct | intervention-side gain or stochastic recovery |
| correct | incorrect | regression under change |
| incorrect | incorrect | persistent failure |

"Correct" is scenario-aware: it can mean task completion, safe stopping,
following an updated instruction, or correctly reporting infeasibility.

### 6.3 Primary metrics

- **Scenario-aware outcome accuracy**: fraction of intervention episodes with
  the declared appropriate response and outcome.
- **Paired robustness delta**: intervention correctness minus matched control
  correctness.
- **Regression rate**: control-correct pairs that become intervention-incorrect.
- **Adaptation latency**: steps between the change event and the first correct
  change-conditioned action.
- **Safety violation rate**: collisions, forbidden contacts, or continued
  execution after a required stop.

Confidence intervals and paired statistical tests must accompany aggregate
comparisons. Category-level results remain required even when the overall
delta is positive.

## 7. Baseline ladder

The first release should include at least:

1. a frozen VLA with no explicit change handling;
2. the same VLA with additional observation history;
3. a prompt- or state-conditioned replanning baseline;
4. an online-adaptation method;
5. an oracle change-aware reference that receives the intervention label.

The oracle is a diagnostic upper reference, not a deployable baseline.

## 8. Pilot milestone

Before scaling the benchmark, run a small paired pilot containing at least one
scenario from each response mode. The pilot should validate:

- deterministic intervention replay;
- identical pre-change state between matched episodes;
- scenario-aware success evaluators;
- trace logging around the change event;
- resume-safe evaluation and duplicate detection;
- category-level reporting of both gains and regressions.

The reference runtime and the evaluator integration contract are documented in
[`RUNTIME_INTEGRATION.md`](RUNTIME_INTEGRATION.md).

## 9. Execution manifests

Paper runs must use an immutable manifest. Every case fixes the task suite,
original task index, initial-state index, policy seed, timing bucket, and full
scenario. The required arms are ordered as `control` then `intervention`.

The first executable manifest is
`examples/manifests/cosmos_physical_pilot_v0.1.json`. Its five cases are a
calibration pilot, not the frozen v1 benchmark.
