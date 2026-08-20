<div align="center">

# LIBERO-MAX: Do Robot Policies Adapt When the World Changes?

**A paired benchmark for dynamic robustness during robot execution**

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-6B7280?style=flat-square)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-available-3B82F6?style=flat-square)](benchmark/max8000)
[![Dynamic cases](https://img.shields.io/badge/Dynamic%20cases-8%2C000-14532D?style=flat-square)](#benchmark)
[![Models](https://img.shields.io/badge/Evaluated%20models-10-C2410C?style=flat-square)](#main-results)
[![Website](https://img.shields.io/badge/Website-project%20page-14532D?style=flat-square)](https://yunbeizhang.github.io/LIBERO-MAX/)
[![CI](https://img.shields.io/github/actions/workflow/status/yunbeizhang/LIBERO-MAX/ci.yml?branch=main&label=tests&style=flat-square)](https://github.com/yunbeizhang/LIBERO-MAX/actions/workflows/ci.yml)

[[Dataset](benchmark/max8000)]
[[Project website](https://yunbeizhang.github.io/LIBERO-MAX/)]
[[Benchmark specification](docs/BENCHMARK_SPEC.md)]
[[Evaluation guide](docs/RUNTIME_INTEGRATION.md)]
[[Paper PDF: coming soon](#citation)]

</div>

<p align="center">
  <img src="assets/figures/benchmark_overview.png" width="100%" alt="LIBERO-MAX benchmark overview">
</p>

Robot benchmarks usually present a fixed world at reset. LIBERO-MAX asks a
different question: **can a policy recover when the world changes after the
robot has already started acting?** Each Dynamic episode is evaluated against
a Base control with the same task, simulator state, policy seed, and action
prefix before the change.

## News

- **2026-08-20:** Expanded the complete 8,000-pair evaluation to **ten**
  VLA/WAM checkpoints and released a model-agnostic dynamic GPU scheduler.
- **2026-08-15:** Released the **LIBERO-MAX dataset** and complete 8,000-pair
  evaluation results across five VLA/WAM models: **π0.5**, **OpenVLA-OFT**,
  **VLA-JEPA**, **Cosmos-Policy**, and **Fast-WAM**.

## To Do

- **Coming soon:** Paper PDF and BibTeX.

## Benchmark

LIBERO-MAX contains **8,000 Dynamic episodes**. During evaluation, every
Dynamic episode receives one matched Base control, producing **8,000 pairs and
16,000 scored rollouts per checkpoint**. The robot begins from the same state
and executes the same action prefix in both arms. Only the Dynamic arm applies
one feasible change during execution.

| Source benchmark | Dynamic cases | Contribution |
| --- | ---: | --- |
| LIBERO-Plus | 5,600 | Controlled visual, scene, and observation variations |
| LIBERO-PRO | 2,400 | Controlled semantic and geometric source variations |
| **LIBERO-MAX** | **8,000** | **Eight changes applied after execution begins** |

The dataset is balanced with 1,000 cases for each event type:

| Family | Online changes |
| --- | --- |
| Observation | Camera shift; sensor corruption |
| Geometry | Target relocation; receptacle relocation |
| Appearance and clutter | Illumination switch; visual theme switch; distractor burst |
| Path constraint | Obstacle insertion |

### Lineage and acknowledgements

LIBERO-MAX is an extension layer rather than a fork of one upstream codebase.
The dynamic event runtime, paired replay protocol, frozen manifests,
aggregation, and validation in `src/libero_max` were developed for this
project. Full evaluation loads the upstream benchmarks as external runtime
dependencies:

| Foundation | Contribution | Paper | Code |
| --- | --- | --- | --- |
| LIBERO | Robot tasks, simulator environment, demonstrations, and success predicates | [paper](https://arxiv.org/abs/2306.03310) | [repository](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO-Plus | Pinned source catalog for 5,600 MAX cases | [paper](https://arxiv.org/abs/2510.13626) | [repository](https://github.com/sylvestf/LIBERO-plus) |
| LIBERO-PRO | Pinned semantic, geometric, and visual substrate for 2,400 MAX cases | [paper](https://arxiv.org/abs/2510.03827) | [repository](https://github.com/Zxy-MLlab/LIBERO-PRO) |

We thank the authors of all three benchmarks for making their datasets and
code available. Exact source revisions and the attribution boundary are
documented in [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md).

<table>
  <tr>
    <td align="center" width="25%"><img src="assets/media/target-relocation.gif" width="220" alt="Target relocation"><br><b>Target relocation</b></td>
    <td align="center" width="25%"><img src="assets/media/receptacle-relocation.gif" width="220" alt="Receptacle relocation"><br><b>Receptacle relocation</b></td>
    <td align="center" width="25%"><img src="assets/media/distractor-burst.gif" width="220" alt="Distractor burst"><br><b>Distractor burst</b></td>
    <td align="center" width="25%"><img src="assets/media/obstacle-insertion.gif" width="220" alt="Obstacle insertion"><br><b>Obstacle insertion</b></td>
  </tr>
</table>

The animations show deterministic MuJoCo intervention states rather than
selected policy successes. All eight visualizations are available in
[`assets/media`](assets/media).

## Main results

Every model is evaluated on the complete set of 8,000 matched pairs. Success
rate is computed over the full denominator, so a policy that never reaches the
event trigger remains an end-to-end failure.

| Family | Model | Base SR ↑ | Dynamic SR ↑ | Gap |
| --- | --- | ---: | ---: | ---: |
| VLA | π0.5 | 79.7 | 65.7 | −13.9 |
| VLA | OpenVLA-OFT | 64.3 | 43.1 | −21.1 |
| VLA | X-VLA | 62.6 | 37.7 | −24.9 |
| VLA | Xiaomi-Robotics-0 | 70.3 | 52.0 | −18.3 |
| VLA | MolmoAct2 | **80.3** | **66.9** | **−13.4** |
| VLA + WM | VLA-JEPA | 73.5 | 54.3 | −19.2 |
| WAM | Cosmos-Policy | 77.4 | 59.3 | −18.2 |
| WAM | Fast-WAM | 42.0 | 24.0 | −18.0 |
| WAM | HiMem-WAM | 73.0 | 57.7 | −15.3 |
| WAM | Light-WAM | 54.8 | 37.3 | −17.5 |

<p align="center">
  <img src="assets/figures/main_results.png" width="95%" alt="Base and Dynamic success rates across ten robot policies">
</p>

All ten policies lose **13.4 to 24.9 success-rate points** when the world
changes during execution. No family dominates across all events. Relocation,
camera shift, and sensor corruption expose the largest shared weaknesses. The
compact source data for the table and plots is available in
[`assets/figures/main_results.json`](assets/figures/main_results.json).

<p align="center">
  <img src="assets/figures/change_type_breakdown.png" width="100%" alt="Success-rate loss by model and online change type">
</p>

The repository focuses on the benchmark contract rather than redistributing
model implementations or weights. These reference adapters demonstrate the
common shard interface used in the reported evaluations:

| Model | Family | Evaluation launcher |
| --- | --- | --- |
| π0.5 | VLA | [`run_openpi_persistent_benchmark.sh`](scripts/run_openpi_persistent_benchmark.sh) |
| OpenVLA-OFT | VLA | [`run_openvla_oft_persistent_benchmark.py`](scripts/run_openvla_oft_persistent_benchmark.py) |
| X-VLA | VLA | [`run_xvla_persistent_shard.py`](scripts/run_xvla_persistent_shard.py) |
| VLA-JEPA | VLA + WM | [`run_vlajepa_persistent_benchmark.py`](scripts/run_vlajepa_persistent_benchmark.py) |
| Cosmos-Policy | WAM | [`run_cosmos_persistent_benchmark.py`](scripts/run_cosmos_persistent_benchmark.py) |
| Fast-WAM | WAM | [`run_fastwam_persistent_benchmark.py`](scripts/run_fastwam_persistent_benchmark.py) |

## Quick start

The benchmark manifest, schema validation, aggregation logic, and unit tests
run without downloading model weights.

```bash
git clone https://github.com/yunbeizhang/LIBERO-MAX.git
cd LIBERO-MAX

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Validate the frozen release and its checksums.
make validate-max8000

# Run the repository test suite.
make test
```

Inspect the complete release through the command line:

```bash
libero-max validate-manifest benchmark/max8000/libero_max_8000.json
```

## Reproducing an evaluation

Model weights and upstream repositories are not redistributed here. A model
adapter consumes one manifest shard, writes terminal `DONE` or `FAILED`
markers, and supports `--resume`. The generic scheduler uses every GPU visible
through `CUDA_VISIBLE_DEVICES` by default and dynamically assigns suite shards
as workers finish:

```bash
python scripts/run_dynamic_benchmark.py \
  benchmark/max8000/libero_max_8000.json \
  --runner scripts/run_xvla_persistent_shard.py \
  --output-root artifacts/xvla \
  -- \
  --lerobot-root /path/to/lerobot \
  --checkpoint /path/to/xvla-libero-checkpoint \
  --query-interval 30
```

Use `--gpus 0,2` to override automatic discovery. A new model can integrate by
implementing the same shard CLI; the scheduler does not import or special-case
the policy. See the [`evaluation guide`](docs/RUNTIME_INTEGRATION.md) for the
adapter contract, simulator requirements, dependency isolation, paired replay
checks, and aggregation. Generated rollouts belong under `artifacts/`, which
is ignored by Git.

## Repository structure

```text
LIBERO-MAX/
├── benchmark/max8000/   # Frozen 8,000-case release and integrity metadata
├── src/libero_max/      # Dynamic interventions, runtime, schemas, aggregation
├── scripts/             # Dataset builders, reference adapters, and GPU scheduler
├── assets/              # Benchmark figures and MuJoCo animations
├── docs/                # Protocol and runtime documentation
├── examples/            # Small manifests and paired-result examples
├── tests/               # Unit, integration, and release-integrity tests
└── index.html            # GitHub Pages project website
```

The public repository intentionally excludes raw rollout traces and generated
paper artifacts. This keeps the code release compact while preserving the
frozen benchmark, the complete evaluation logic, and every launcher required
to reproduce the reported experiments.

## Citation

The paper PDF and verified BibTeX entry will be added with the manuscript
release. Until then, please cite the repository and dataset release:

```text
LIBERO-MAX Dataset
LIBERO-MAX: Do Robot Policies Adapt When the World Changes?
https://github.com/yunbeizhang/LIBERO-MAX
```
