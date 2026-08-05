# LIBERO-MAX v1 Results

This directory contains the complete paper evaluation for the frozen v1 Core
benchmark. All reported units are deterministic matched control/intervention
pairs. Missing or invalid pairs are never imputed as policy failures.

## Evaluation coverage

| Evaluation | Model | Valid pairs | Trigger coverage |
| --- | --- | ---: | ---: |
| Physical Core | Cosmos Policy Predict2-2B | 1,335 / 1,335 | 100% |
| Physical Core | pi0.5-LIBERO | 1,335 / 1,335 | 100% |
| Intent Core | Cosmos Policy Predict2-2B | 96 / 96 | 100% |
| Intent Core | pi0.5-LIBERO | 96 / 96 | 100% |
| Balanced q5 ablation | Cosmos Policy Predict2-2B | 180 / 180 | 100% |
| Balanced notified-q16 ablation | Cosmos Policy Predict2-2B | 180 / 180 | 100% |

This is 3,222 independently executed pairs / 6,444 rollouts. The published
q16 balanced subset is materialized from the complete Cosmos Physical Core run
and is not counted as another execution.

## Main results

| Model and track | Control | Changed outcome | Paired delta (95% CI) |
| --- | ---: | ---: | ---: |
| Cosmos Physical | 98.7% | 78.1% | -20.7 [-22.9, -18.4] |
| pi0.5 Physical | 96.6% | 82.6% | -13.9 [-16.0, -11.9] |
| Cosmos Intent | 100.0% | 13.5% | -86.5 [-92.7, -79.2] |
| pi0.5 Intent | 96.9% | 15.6% | -81.2 [-88.5, -72.9] |

On Physical Core, pi0.5 exceeds Cosmos by 4.6 points on changed-arm success
(95% CI [2.4, 6.8], exact McNemar p=6.70e-5) and has a 6.7-point smaller
paired robustness gap (95% CI [4.4, 9.1]). On Intent Core, the 2.1-point
changed-outcome difference is not significant (95% CI [-3.1, 7.3], p=0.6875).

## Ablations

| Cosmos condition | Control | Changed outcome | Event-to-query steps, mean / median |
| --- | ---: | ---: | ---: |
| q16 | 97.8% | 72.2% | 7.94 / 8.5 |
| q5 | 99.4% | 75.0% | 1.98 / 2.0 |
| notified q16 | 98.9% | 66.1% | 7.88 / 8.0 |

Relative to q16, q5 changes intervention success by +2.8 points (95% CI
[-2.2, 7.8], p=0.3833): exposure is much shorter, but the success gain is not
statistically reliable. Generic notification changes success by -6.1 points
(95% CI [-10.6, -1.7], p=0.0127).

The q5 evaluator executes the first five actions predicted in each Cosmos
chunk. This prefix contract is implemented by `retain_action_prefix` and is
covered by the repository tests.

## Files

- `runs/*/benchmark_summary.json` contains aggregate coverage, metrics,
  uncertainty, and outcome counts for one run.
- `runs/*/paired_results.jsonl` contains the auditable pair-level records.
- `tables/main/` contains the four-run Physical/Intent tables.
- `tables/ablation/` contains the matched q16/q5/notification tables.
- `SHA256SUMS` pins every published result artifact.

The paper-ready narrative and claim boundaries are in
[`../../paper/main.md`](../../paper/main.md). Raw videos and per-step simulator
traces are omitted from Git because of size; the pair-level records retain the
resolved case keys, outcomes, triggers, and diagnostics used to regenerate all
tables.
