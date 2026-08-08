# LIBERO-MAX-8000 candidate artifacts

This directory contains a deterministic candidate, not a frozen release.

- `pro_source_lock.json`: upstream code/data revisions and category boundary;
- `pro_task_catalog.json`: 400 selected LIBERO-PRO task variants;
- `libero_max_pro_hard_2400.json`: balanced PRO-Hard paired manifest;
- `libero_max_8000.json`: exact Base-5600 + PRO-Hard-2400 union;
- `release_summary.json`: candidate counts and outstanding release gate.
- `preflight_screening_round1.json`: first real-MuJoCo rejection matrix and
  deterministic replacement accounting;
- `pro_rejection_ledger.json`: all 1,710 infeasible candidate configurations;
- `pro_physical_preflight.json`: complete 2,400/2,400 real-MuJoCo report with
  per-case PRO component audits and pre/post visibility checks;
- `SHA256SUMS`: checksums for the deterministic candidate artifacts.

All 2,400 PRO-Hard configurations now pass real-MuJoCo preflight. The candidate
is promoted to `benchmark/max8000/` only after the PRO-aware config runtime is
published or vendored and the version 3.0.0 freeze audit succeeds. See
`docs/MAX_PRO_HARD_DESIGN.md`.
