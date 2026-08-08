# MAX-8000 paper experiment matrix

This file freezes the evaluation matrix and evidence gates used for the
LIBERO-MAX paper-scale result bundle. Selection is outcome-independent. No case
is removed because a model fails the control task, fails to approach the
trigger, or reaches the trigger too late to issue another policy query.

## Headline physical evaluation

| Evaluation | Models | Cases per model | Query interval | Purpose |
| --- | --- | ---: | ---: | --- |
| MAX-Base | Cosmos Policy Predict2-2B; pi0.5-LIBERO | 5,600 | native q16 / q5 | Dynamic events on balanced LIBERO-Plus tasks |
| MAX-PRO-Hard | Cosmos Policy Predict2-2B; pi0.5-LIBERO | 2,400 | native q16 / q5 | Dynamic events composed with ten static LIBERO-PRO substrate categories |
| MAX-8000 | exact Base + PRO union | 8,000 | native q16 / q5 | Full paper-scale end-to-end result |

Base and PRO-Hard results are reported separately before the combined
micro-average. Each case contains a control and intervention arm, so MAX-8000
requires 16,000 simulator rollouts per evaluated model.

## Frozen cross-model comparison

Cosmos Policy, pi0.5-LIBERO, FastWAM-LIBERO, and LingBot-VA are evaluated on
the same 800-case MAX-PRO subset at q16. The subset contains ten PRO substrate
categories x eight event types x two draws x five cases. It is frozen before
model outcomes are observed and is used for paired model comparisons, not as a
replacement for the two-model full MAX-8000 evaluation.

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
any versioned correction. Failure evidence is normalized by model coverage so
the 800-case four-model subset is not preferentially ranked simply because it
has more evaluated policies.

## Media and publication gate

The release includes eight real-MuJoCo before/after GIFs, an overview image,
and one deterministic replay of a response-evaluable recorded Cosmos action
trace with event-step verification. The replay selector prefers successful
target/receptacle relocation cases and records the exact selected case ID.
Final README results are generated only when every listed run is
execution-complete. The compact bundle contains Markdown and LaTeX tables,
JSON summaries, full-denominator rows, figures, the human-review queue, run
provenance, and SHA-256 checksums.
