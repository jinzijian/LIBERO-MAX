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
4. preflight every unique Core candidate configuration in real MuJoCo;
5. require nonzero visual change, visible images, finite simulator state,
   baseline-relative stability, declared support, and no new unrelated contact;
6. if any initial state fails, exclude that entire task--change--draw
   configuration from every initial state and policy seed;
7. rerun the complete physical preflight on the filtered Core;
8. verify that Core contains every physical scenario once and Full contains the
   identical scenarios once for each of three policy seeds;
9. freeze only a complete report and write SHA-256 checksums.

The release directory contains:

```text
benchmark/v1/
├── task_catalog.json
├── core.json
├── full.json
├── physical_preflight.json
├── feasibility_filter.json
├── release_summary.json
└── SHA256SUMS
```

`release_summary.json` is the authoritative final count. v1.0.0 contains 40
tasks across four suites, 177 task--change cells, 1,335 Core matched pairs, and
4,005 Full matched pairs. The final physical preflight passed 1,335/1,335; 62
candidate task--change--draw configurations are recorded as excluded.

Target and receptacle relocation never sample a direction during evaluation.
For each task, both distances use the same calibrated unit direction, and the
manifest stores the fully resolved displacement. If no common direction passes
all required states and distances, that task--change-type cell is excluded
before model evaluation rather than recorded as a policy failure.

This release gate validates the resolved simulator transitions, not policy
success. Evaluation must separately report control completion, trigger firing,
paired intervention coverage, and any invalid rollout.
