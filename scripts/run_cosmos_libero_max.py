#!/usr/bin/env python3
"""Run upstream Cosmos Policy with LIBERO-MAX environment hooks."""

import os
from pathlib import Path

from libero.libero import benchmark

from libero_max.cosmos_integration import install_cosmos_hooks
from libero_max.scenario import load_scenarios, validate_scenario_collection


def install_subset_benchmark(suite_name: str, task_index: int) -> None:
    original_class = benchmark.get_benchmark_dict()[suite_name]

    class SubsetBenchmark(original_class):
        def __init__(self, task_order_index: int = 0):
            super().__init__(task_order_index=task_order_index)
            original_count = self.n_tasks
            if not 0 <= task_index < original_count:
                raise ValueError(
                    "task index %d outside [0, %d)" % (task_index, original_count)
                )
            self.tasks = [self.tasks[task_index]]
            self.n_tasks = 1
            print(
                "[LIBERO-MAX] %s: original task index %d (1/%d)"
                % (suite_name, task_index, original_count),
                flush=True,
            )

    benchmark.BENCHMARK_MAPPING[suite_name] = SubsetBenchmark


def main() -> None:
    suite_name = os.environ.get("LIBERO_MAX_SUITE", "libero_object")
    task_index = int(os.environ.get("LIBERO_MAX_TASK_INDEX", "0"))
    arm = os.environ.get("LIBERO_MAX_ARM", "intervention")
    init_state_index = int(os.environ.get("LIBERO_MAX_INIT_STATE_INDEX", "0"))
    scenario_path = Path(
        os.environ.get(
            "LIBERO_MAX_SCENARIO_FILE",
            "examples/scenarios/cosmos_camera_shift_chunk16.json",
        )
    )
    trace_path = Path(
        os.environ.get("LIBERO_MAX_TRACE_PATH", "artifacts/cosmos_trace.jsonl")
    )
    control_trace_raw = os.environ.get("LIBERO_MAX_CONTROL_TRACE_PATH")
    control_trace_path = Path(control_trace_raw) if control_trace_raw else None
    scenarios = load_scenarios([scenario_path])
    errors = validate_scenario_collection(scenarios)
    if errors:
        raise ValueError("; ".join(errors))
    if len(scenarios) != 1:
        raise ValueError("Cosmos launcher requires exactly one scenario")

    install_subset_benchmark(suite_name, task_index)
    from cosmos_policy.experiments.robot.libero import run_libero_eval

    install_cosmos_hooks(
        run_libero_eval=run_libero_eval,
        scenario=scenarios[0],
        arm=arm,
        trace_path=trace_path,
        original_task_index=task_index,
        init_state_index=init_state_index,
        control_trace_path=control_trace_path,
    )
    run_libero_eval.eval_libero()


if __name__ == "__main__":
    main()
