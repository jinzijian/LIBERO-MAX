# MAX-PRO-Hard-2400 design

Status: candidate pending complete real-MuJoCo preflight

## Scientific role

LIBERO-PRO tests generalization when a perturbed world is presented before the
policy acts. LIBERO-MAX tests adaptation when another exogenous change occurs
after the policy has started acting. MAX-PRO-Hard composes the two axes without
conflating their control arms:

1. both arms start from the same LIBERO-PRO BDDL and initialization state;
2. both arms receive the same perturbed instruction and exact pre-event action
   chunks;
3. only the intervention arm receives one MAX event at target proximity;
4. static PRO failure, trigger coverage, and post-trigger adaptation remain
   separately measurable.

The Base-5600 and PRO-Hard-2400 scores are always reported separately. A single
MAX-8000 micro-average is supplementary and cannot replace the two track
results or a category-macro average.

## Size and balance

| Component | Construction | Matched pairs | Rollouts/model |
| --- | --- | ---: | ---: |
| MAX-Base | 7 Plus categories x 800 | 5,600 | 11,200 |
| MAX-PRO-Hard | 10 PRO categories x 8 changes x 2 draws x 15 | 2,400 | 4,800 |
| MAX-8000 | exact union | 8,000 | 16,000 |

The PRO source contains 400 distinct selected BDDL task variants: 40 per
category across `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. Each joint category/change/draw cell contains 15 unique
`(task, init_state)` configurations. A BDDL task can appear under multiple MAX
events; therefore 2,400 pairs must not be described as 2,400 unique tasks.

The ten substrate categories are:

1. semantic instruction paraphrase;
2. object perturbation;
3. initial position/relation perturbation;
4. task/goal perturbation;
5. visual noise and glare;
6. camera viewpoint;
7. object texture;
8. view occlusion;
9. object shape;
10. initial pose, position, and yaw.

For the six config-driven robustness categories, loading a BDDL is not enough.
The evaluator applies the pinned `:perturbation_config` after restoring the
frozen state, and transforms every policy observation when the category calls
for observation noise. `view_occlusion` adds one free object joint; its shared
original init state is mapped into the extended model by joint name, while the
new occluder keeps its deterministic reset pose before the configured
front-occlusion placement is applied. PRO-only occluders are reserved and can
never be sampled again as MAX distractors.

The eight MAX events remain illumination switch, camera pose/FOV shift, visual
theme switch, sensor-corruption onset, target relocation, receptacle
relocation, five-object distractor burst, and obstacle insertion.

## Upstream source boundary

The candidate pins:

- code: `Zxy-MLlab/LIBERO-PRO` commit
  `eafdb809426b13153aa1e4c42d6601844217dfec`;
- dataset: `zhouxueyang/LIBERO-Pro` revision
  `c86fc3b8293185a6f373677018ff3e37f8391602`;
- dataset license: CC-BY-4.0.

The current real-MuJoCo candidate uses the PRO-aware runtime commit
`2b910b5b5f53016bef9907632f6f840f1ce2229c`, which provides the BDDL config
parser and runtime perturbation hooks. That dependency must be published or
vendored before the version 3.0.0 freeze; the setup script refuses a plain PRO
checkout that lacks those modules.

The upstream seven-case `runtime_object_move` category is intentionally
excluded because it already introduces a near-grasp runtime change. Treating
it as a static substrate would destroy the identical pre-event causal
contract. The advertised environment category is also excluded from this
revision because complete environment BDDL/init artifacts are not present in
the pinned public dataset. It can be added only after a reproducible frozen
artifact release and independent MuJoCo validation.

## Reproducible construction

Download and configure the pinned external artifacts:

```bash
python scripts/setup_libero_pro_substrate.py \
  --libero-pro-root /path/to/LIBERO-PRO \
  --dataset-root /path/to/libero-pro-data \
  --config-dir /path/to/libero-pro-config
```

Build the catalog and manifests:

```bash
PYTHONPATH=src python scripts/build_libero_pro_catalog.py \
  --dataset-root /path/to/libero-pro-data \
  --libero-task-map /path/to/LIBERO-PRO/libero/libero/benchmark/libero_suite_task_map.py \
  --source-revision c86fc3b8293185a6f373677018ff3e37f8391602 \
  --output benchmark/max8000_candidate/pro_task_catalog.json

PYTHONPATH=src python scripts/build_libero_max_8000.py \
  --base-manifest benchmark/max5600/libero_max_5600.json \
  --pro-catalog benchmark/max8000_candidate/pro_task_catalog.json \
  --pro-output benchmark/max8000_candidate/libero_max_pro_hard_2400.json \
  --combined-output benchmark/max8000_candidate/libero_max_8000.json \
  --summary benchmark/max8000_candidate/release_summary.json
```

## Release gates

The candidate becomes version 3.0.0 only when:

1. the catalog contains exactly 10 categories and 400 source-locked variants;
2. all 160 category/change/draw cells contain exactly 15 pairs;
3. every BDDL and init path resolves below the configured PRO data roots;
4. all init-state indices resolve in the official artifacts;
5. all 2,400 resolved transitions pass real-MuJoCo geometry, support, contact,
   stability, and visibility checks;
6. rejected candidates are deterministically replaced within the same cell and
   the complete 2,400-case preflight is rerun;
7. rollout traces prove identical control/intervention initial states and
   pre-event action chunks;
8. infrastructure gaps are repaired rather than charged to the model;
9. the freeze audit creates versioned manifests and SHA-256 checksums.
10. the pinned PRO-aware runtime is publicly retrievable or vendored with the
    release.

Run the remote preflight with the PRO implementation and data configuration:

```bash
LIBERO_PRO_DIR=/path/to/LIBERO-PRO \
LIBERO_PRO_CONFIG=/path/to/libero-pro-config \
bash scripts/run_max_pro_preflight.sh \
  benchmark/max8000_candidate/libero_max_pro_hard_2400.json \
  artifacts/max8000/pro_preflight
```

Only after a complete passing report may
`scripts/freeze_libero_max_8000.py` create `benchmark/max8000/`.
