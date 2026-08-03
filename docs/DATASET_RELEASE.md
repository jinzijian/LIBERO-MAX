# LIBERO-MAX v1 Test-Set Release Contract

The versioned test sets are generated artifacts, not hand-edited JSON files.
Run the complete pipeline only on the pinned LIBERO / robosuite environment:

```bash
bash scripts/run_v1_dataset_build.sh
```

The pipeline performs these gates in order:

1. rebuild the 40-task catalog from installed BDDL files;
2. choose one fixed relocation direction per eligible task and require both
   6 cm and 12 cm to pass all three initial states;
3. generate Core and Full candidate manifests;
4. preflight every unique Core physical configuration in real MuJoCo;
5. require nonzero visual change, visible images, finite simulator state,
   baseline-relative stability, declared support, and no new unrelated contact;
6. verify that Core contains every physical scenario once and Full contains the
   identical scenarios once for each of three policy seeds;
7. freeze only a complete report and write SHA-256 checksums.

The release directory contains:

```text
benchmark/v1/
├── task_catalog.json
├── core.json
├── full.json
├── physical_preflight.json
├── release_summary.json
└── SHA256SUMS
```

`release_summary.json` is the authoritative final count. Design-document
counts are pre-calibration upper bounds until the release audit succeeds.

Target and receptacle relocation never sample a direction during evaluation.
For each task, both distances use the same calibrated unit direction, and the
manifest stores the fully resolved displacement. If no common direction passes
all required states and distances, that task--change-type cell is excluded
before model evaluation rather than recorded as a policy failure.
