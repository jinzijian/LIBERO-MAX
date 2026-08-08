# Human feasibility secondary review

The human-review queue prioritizes cases that deserve expert teleoperation. It
does not remove cases from LIBERO-MAX and is not itself evidence that a task is
infeasible. Model failure, including failure in every evaluated control arm,
is a triage signal rather than a benchmark defect.

## Frozen protocol

For every queued case, use the released manifest entry, source-locked
LIBERO/LIBERO-PRO runtime, initial state, intervention draw, trigger, and action
interface. The operator may see the same two camera observations available to
the evaluated policy and must not edit the MuJoCo state or event placement.

Run three independent teleoperation attempts. In every attempt, first execute
the original task until the frozen proximity trigger applies the intervention,
then continue toward the original goal. Record whether the trigger fired,
whether the simulator remained valid, and whether the original goal completed.

## Labels

- `human_feasible`: at least one of three attempts completes the task after the
  event without a simulator or safety-contract violation.
- `inconclusive`: the UI, operator, recording, or trigger failed to provide
  three interpretable attempts.
- `benchmark_defect_candidate`: zero of three expert attempts succeeds and the
  traces show a repeatable geometry, reachability, visibility, or simulator
  problem. This label requires a second reviewer before any release decision.
- `confirmed_benchmark_defect`: two reviewers reproduce the defect and a
  corrected configuration passes the same physical preflight and teleoperation
  protocol. Changes must be versioned; historical results are never silently
  rewritten.

The generated CSV contains blank attempt, success, label, reviewer, and notes
columns for the review team. A successful model control is retained as
counterevidence to base-task infeasibility; a successful model intervention is
stronger counterevidence because it completes the exact changed episode.
Neither replaces the direct positive check from a successful human
intervention rollout.

Model-failure evidence is normalized by the number of models evaluated for a
case. This prevents the frozen 800-case four-model subset from receiving a
higher priority merely because it has two additional model outcomes; ranking
depends on failure rate and the physical/task risk signals instead.
