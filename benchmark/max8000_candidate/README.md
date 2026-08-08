# LIBERO-MAX-8000 candidate artifacts

This directory contains a deterministic candidate, not a frozen release.

- `pro_source_lock.json`: upstream code/data revisions and category boundary;
- `pro_task_catalog.json`: 400 selected LIBERO-PRO task variants;
- `libero_max_pro_hard_2400.json`: balanced PRO-Hard paired manifest;
- `libero_max_8000.json`: exact Base-5600 + PRO-Hard-2400 union;
- `release_summary.json`: candidate counts and outstanding release gate.
- `SHA256SUMS`: checksums for the deterministic candidate artifacts.

The candidate is promoted to `benchmark/max8000/` only after all 2,400
PRO-Hard configurations pass real-MuJoCo preflight and the version 3.0.0 freeze
audit succeeds. The PRO-aware config runtime must also be published or vendored
before release. See `docs/MAX_PRO_HARD_DESIGN.md`.
