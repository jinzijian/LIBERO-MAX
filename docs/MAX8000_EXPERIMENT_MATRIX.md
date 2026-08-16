# MAX-8000 paper experiment matrix

This file freezes the evaluation matrix and evidence gates used for the
LIBERO-MAX paper-scale result bundle. Selection is outcome-independent. No case
is removed because a model fails the control task, fails to approach the
trigger, or reaches the trigger too late to issue another policy query.

## Headline physical evaluation

| Evaluation | Models | Cases per model | Query interval | Purpose |
| --- | --- | ---: | ---: | --- |
| MAX-Base | Cosmos Policy Predict2-2B; pi0.5-LIBERO; FastWAM-LIBERO; LingBot-VA | 5,600 | native q16 / q5 | Dynamic events on balanced LIBERO-Plus tasks |
| MAX-PRO-Hard | same four models | 2,400 | native q16 / q5 | Dynamic events composed with ten static LIBERO-PRO substrate categories |
| MAX-8000 | same four models on the exact Base + PRO union | 8,000 | native q16 / q5 | Full paper-scale end-to-end and cross-model result |

Base and PRO-Hard results are reported separately before the combined
micro-average. Each case contains a control and intervention arm, so MAX-8000
requires 16,000 simulator rollouts per evaluated model. All four models cover
all 8,000 pairs. Cosmos, FastWAM, and LingBot use their native q16 commitment;
pi0.5 uses its native q5 action horizon. Query interval is reported with every
result and is not changed beyond the checkpoint's returned action chunk.

## Frozen cross-model comparison

Cosmos Policy, pi0.5-LIBERO, FastWAM-LIBERO, and LingBot-VA are evaluated on
the same full 8,000-case benchmark at their native query intervals. The frozen 800-case
MAX-PRO subset contains ten PRO substrate categories x eight event types x two
draws x five cases. It remains an outcome-independent early comparison and is
reused as coverage inside the full run, but it is not the paper's cross-model
denominator. The missing 1,600 PRO cases and all 5,600 Base cases are evaluated
for every model.

## Existing response and evaluator ablations

- The 96-case Intent Core evaluates target update, receptacle update, and task
  cancellation with response-aware correctness for Cosmos and pi0.5.
- A balanced 180-case physical subset compares Cosmos q16, Cosmos q5, and
  evaluator-provided generic event notification at q16.

These are reported as separate tracks and ablations. They are not pooled into
the physical MAX-8000 denominator.

## Required statistics

Every physical run publishes:

1. control and intervention success on the full frozen denominator;
2. paired robustness delta and paired outcome decomposition;
3. trigger coverage and post-event response-query coverage separately;
4. response-conditioned paired delta with bootstrap confidence interval;
5. exact McNemar test from intervention regressions and recoveries, with Holm
   correction across the six four-model comparisons;
6. breakdowns by event, severity, draw, task suite, and PRO substrate;
7. both full micro-averages and unweighted substrate-category macro-averages;
8. paired model comparisons on the exact common case IDs.

Infrastructure or trace-integrity gaps remain missing and block publication.
`trigger_unreached` is retained as a pre-intervention policy failure.
`response_query_unreached` is retained in end-to-end outcomes but excluded from
response-conditioned diagnostics because no post-event policy output exists.

## Human feasibility secondary review

All 8,000 manifest entries already pass automated physical preflight. After
model evaluation, a deterministic risk score prioritizes up to 300 cases for
expert teleoperation. Model failures are triage signals only and never remove
cases from reported denominators. Reviewers make three attempts per case; one
success establishes positive feasibility evidence. Zero of three creates a
benchmark-defect candidate that requires a second independent reviewer before
any versioned correction. Every model has the same full coverage, so
model-failure risk signals are computed directly on the common 8,000-case
denominator.

## Media and publication gate

The release includes eight real-MuJoCo before/after GIFs, an overview image,
and one deterministic replay of a response-evaluable recorded Cosmos action
trace with event-step verification. The replay selector prefers successful
target/receptacle relocation cases and records the exact selected case ID.
Final README results are generated only when every listed run is
execution-complete. The compact bundle contains Markdown and LaTeX tables,
JSON summaries, full-denominator rows, figures, the human-review queue, run
provenance, and SHA-256 checksums.
