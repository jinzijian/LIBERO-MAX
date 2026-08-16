# Test-suite map

The tests are part of the benchmark release, not generated rollout data. They
protect the frozen dataset and the evaluation contract from accidental changes.
The suite currently contains 203 automated checks in five groups.

| Group | Representative files | What it protects | Keep in a public release? |
| --- | --- | --- | --- |
| Frozen release integrity | `test_max8000_release.py`, `test_manifest.py`, `test_schema_sync.py`, `test_release.py` | 8,000 unique cases, balanced event counts, source locks, schema compatibility, and checksums | Yes, essential |
| Dynamic runtime | `test_runtime.py`, `test_libero_backend.py`, `test_pro_runtime.py`, `test_env_factory.py`, `test_mujoco_geometry.py` | Trigger timing, intervention application, paired state, renderer stability, and physical validity | Yes, essential |
| Paired scoring and aggregation | `test_results.py`, `test_compose_paired_run.py`, `test_aggregate_end_to_end.py`, `test_model_run_provenance.py` | Complete denominators, Base versus Dynamic pairing, resume behavior, and run provenance | Yes, essential |
| Model launcher contracts | `test_launchers.py`, `test_fastwam_adapter.py`, `test_model_launcher_isolation.py`, `test_full_model_queue.py` | Checkpoint arguments, environment isolation, sharding, and model-specific integration assumptions | Yes, while those launchers are supported |
| Reporting and media | `test_paper_tables.py`, `test_paper_figures.py`, `test_render_media.py`, `test_rollout_replay_media.py` | Published tables, figures, and visual examples remain synchronized with verified results | Useful, but separable from a minimal runtime package |

For the current research release, keeping the complete suite is preferable: it
is small, runs without downloading model weights, and allows users to verify
the benchmark before allocating expensive GPU time. If the repository is later
split into a lightweight runtime package and a paper-artifact package, the last
group can move with the reporting tools.

Run everything with:

```bash
make test
```
