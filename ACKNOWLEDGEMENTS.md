# Acknowledgements and provenance

LIBERO-MAX is an extension layer for evaluating dynamic changes during robot
execution. Its paired protocol, online intervention runtime, frozen manifests,
release validation, aggregation, and model-adapter orchestration were developed
for this project. The repository does not vendor the upstream benchmark or
model repositories listed below. They are installed separately when running a
full simulator evaluation and retain their respective licenses.

## Benchmark foundations

| Project | Role in LIBERO-MAX | Paper | Code |
| --- | --- | --- | --- |
| LIBERO | Provides the robot tasks, demonstrations, simulator environment, and standard task-success predicates. | [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310) | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO-Plus | Provides the pinned source catalog from which 5,600 cases are selected. LIBERO-MAX converts those static variations into online events while preserving the source task identity. | [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models](https://arxiv.org/abs/2510.13626) | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) |
| LIBERO-PRO | Provides the pinned semantic, geometric, and visual substrate cases from which 2,400 cases are selected. LIBERO-MAX applies its dynamic event protocol on top of those source cases. | [LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization](https://arxiv.org/abs/2510.03827) | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) |

The frozen MAX-8000 release records the exact source revisions used for its two
components:

- LIBERO-Plus code revision `4976dc30028e805ff8094b55501d532c48fec182`;
- LIBERO-PRO dataset revision `c86fc3b8293185a6f373677018ff3e37f8391602`;
- LIBERO-PRO code revision inspected for compatibility
  `eafdb809426b13153aa1e4c42d6601844217dfec`.

The corresponding machine-readable provenance is stored in
[`benchmark/max8000/release_summary.json`](benchmark/max8000/release_summary.json)
and [`benchmark/max8000/pro_source_lock.json`](benchmark/max8000/pro_source_lock.json).

## Evaluation codebases

The released launchers interface with the official or project-provided
implementations of π0.5, OpenVLA-OFT, VLA-JEPA, Cosmos-Policy, and Fast-WAM.
Model weights and upstream model repositories are not redistributed here. We
thank all model authors for releasing the checkpoints and evaluation code that
make matched comparison possible.

## Attribution boundary

Using LIBERO-MAX does not replace citation of its benchmark foundations. Work
that evaluates on the MAX-8000 release should cite LIBERO-MAX together with
LIBERO, LIBERO-Plus, and LIBERO-PRO as appropriate for the source cases used.
Users are also responsible for following the licenses of the simulator,
benchmark, dataset, and model repositories installed in their environment.
