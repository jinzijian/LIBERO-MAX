"""Resolve base LIBERO and file-backed LIBERO-PRO task variants."""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, MutableMapping, Optional, Tuple


def variant_path(root: str, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe substrate variant path: %s" % relative)
    return Path(root).joinpath(*parts)


def _load_trusted_initial_states(torch_module: Any, path: Path) -> Any:
    """Load a source-locked LIBERO-PRO initial-state artifact.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    These benchmark files contain NumPy arrays rather than model weights, so
    the restricted loader rejects them. Their SHA-256 digests are locked in
    the release provenance; explicitly use the legacy loader for this trusted
    local artifact while retaining compatibility with older PyTorch releases.
    """

    try:
        return torch_module.load(path, weights_only=False)
    except TypeError:
        return torch_module.load(path)


def load_case_task(
    case: Dict[str, Any],
    benchmark: Any,
    suite_cache: Optional[MutableMapping[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Return the task descriptor and initial-state array for one case.

    Normal cases use the registered LIBERO suite.  LIBERO-PRO cases keep the
    base suite name in the manifest for action normalization and horizons, but
    load their BDDL and initialization files from explicit, source-locked
    paths below LIBERO's configured data roots.
    """

    variant = case.get("substrate_variant")
    if variant is None:
        suite_name = case["task_suite_name"]
        if suite_cache is not None and suite_name in suite_cache:
            suite = suite_cache[suite_name]
        else:
            suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
            if suite_cache is not None:
                suite_cache[suite_name] = suite
        task_index = case["task_index"]
        if not 0 <= task_index < suite.n_tasks:
            raise IndexError("task index is outside suite")
        return suite.get_task(task_index), suite.get_task_init_states(task_index)

    if variant.get("benchmark") != "LIBERO-PRO":
        raise ValueError("unsupported substrate variant benchmark")
    from libero.libero import get_libero_path
    import torch

    bddl_path = variant_path(get_libero_path("bddl_files"), variant["bddl_file"])
    init_path = variant_path(
        get_libero_path("init_states"), variant["init_states_file"]
    )
    if not bddl_path.is_file():
        raise FileNotFoundError(str(bddl_path))
    if not init_path.is_file():
        raise FileNotFoundError(str(init_path))
    relative_bddl = PurePosixPath(variant["bddl_file"])
    task = benchmark.Task(
        name=case.get("task_name", relative_bddl.stem),
        language=variant["language"],
        problem="Libero",
        problem_folder=str(relative_bddl.parent),
        bddl_file=relative_bddl.name,
        init_states_file=PurePosixPath(variant["init_states_file"]).name,
    )
    initial_states = _load_trusted_initial_states(torch, init_path)
    return task, initial_states
