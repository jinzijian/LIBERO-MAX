"""Non-invasive LIBERO-MAX hooks for the upstream Cosmos Policy evaluator."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .runtime import INTENT_OPERATIONS, InterventionRuntime, TriggerContext


class CosmosIntegrationError(RuntimeError):
    """Raised when Cosmos integration invariants are not satisfied."""


def _state_digest(state: Any) -> str:
    if hasattr(state, "tobytes"):
        payload = state.tobytes()
    else:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    return hashlib.sha256(payload).hexdigest()


def _mean_absolute_pixel_delta(before: Any, after: Any) -> Optional[float]:
    if before is None or after is None:
        return None
    import numpy as np

    before_array = np.asarray(before, dtype=np.int16)
    after_array = np.asarray(after, dtype=np.int16)
    if before_array.shape != after_array.shape:
        raise CosmosIntegrationError(
            "pre/post intervention image shapes differ: %s vs %s"
            % (before_array.shape, after_array.shape)
        )
    return float(np.mean(np.abs(after_array - before_array)))


class CosmosInterventionEnv:
    """Delegate LIBERO env calls while injecting one chunk-aligned change."""

    def __init__(
        self,
        env: Any,
        task_description: str,
        scenario: Dict[str, Any],
        arm: str,
        trace_path: Path,
        original_task_index: int,
        init_state_index: int = 0,
        backend: Any = None,
        warmup_steps: int = 10,
        primary_image_key: Optional[str] = "agentview_image",
    ):
        if arm not in {"control", "intervention"}:
            raise CosmosIntegrationError("arm must be control or intervention")
        if backend is None:
            from .libero_backend import LiberoMujocoBackend

            backend = LiberoMujocoBackend(env)

        self._env = env
        self.task_description = task_description
        self.scenario = scenario
        self.arm = arm
        self.trace_path = Path(trace_path)
        self.original_task_index = original_task_index
        self.init_state_index = init_state_index
        self.backend = backend
        self.runtime = InterventionRuntime(scenario, backend)
        self.warmup_steps = warmup_steps
        self.primary_image_key = primary_image_key
        self.query_interval = None
        self.max_policy_steps = None
        self.policy_seed = None
        self.task_suite_name = None
        self.episode_index = -1
        self.total_env_steps = 0
        self.init_state_sha256 = None
        self.events = []
        self.policy_queries = []
        self.setup_events = []
        self.trigger_observation = None
        self.executed_actions: List[Dict[str, Any]] = []
        self.original_goal_completed_after_event = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def configure_episode(
        self,
        task_suite_name: str,
        policy_seed: int,
        query_interval: int,
        max_policy_steps: int,
    ) -> None:
        if query_interval <= 0 or max_policy_steps <= 0:
            raise CosmosIntegrationError("Cosmos step configuration must be positive")
        self.task_suite_name = task_suite_name
        self.policy_seed = policy_seed
        self.query_interval = query_interval
        self.max_policy_steps = max_policy_steps

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self._env.reset(*args, **kwargs)
        self.episode_index += 1
        self.total_env_steps = 0
        self.init_state_sha256 = None
        self.events = []
        self.policy_queries = []
        self.setup_events = []
        self.trigger_observation = None
        self.executed_actions = []
        self.original_goal_completed_after_event = False
        self.runtime.reset(self.task_description)
        return result

    def set_init_state(self, initial_state: Any) -> Any:
        self.init_state_sha256 = _state_digest(initial_state)
        observation = self._env.set_init_state(initial_state)
        self.setup_events = self.runtime.apply_setup()
        if self.setup_events:
            observation = self.backend.refresh_observation()
        return observation

    def _capture_primary_image(self, observation: Any) -> Any:
        if self.primary_image_key is None or not isinstance(observation, dict):
            return None
        image = observation.get(self.primary_image_key)
        return image.copy() if hasattr(image, "copy") else image

    def step(self, action: Any) -> Any:
        observation, reward, done, info = self._env.step(action)
        original_done = bool(done)
        self.total_env_steps += 1
        policy_step = max(0, self.total_env_steps - self.warmup_steps)

        trigger = self.scenario["trigger"]
        trigger_events = frozenset()
        proximity_distance = None
        if trigger["type"] == "on_proximity" and policy_step > 0:
            proximity_distance = self.backend.distance_to_entity(
                observation, trigger["value"]
            )
            if proximity_distance <= trigger["distance_m"]:
                trigger_events = frozenset({"proximity:%s" % trigger["value"]})
                if self.trigger_observation is None:
                    self.trigger_observation = {
                        "type": "on_proximity",
                        "entity": trigger["value"],
                        "threshold_m": trigger["distance_m"],
                        "distance_m": proximity_distance,
                        "policy_step": policy_step,
                    }

        can_apply = (
            self.arm == "intervention"
            and not done
            and self.max_policy_steps is not None
            and policy_step > 0
            and (
                bool(trigger_events)
                or (
                    trigger["type"] != "on_proximity"
                    and self.query_interval is not None
                    and policy_step % self.query_interval == 0
                )
            )
        )
        if can_apply:
            before_image = self._capture_primary_image(observation)
            event = self.runtime.maybe_apply(
                TriggerContext(
                    step=policy_step,
                    max_steps=self.max_policy_steps,
                    events=trigger_events,
                )
            )
            if event is not None:
                if self.scenario["change"]["operation"] not in INTENT_OPERATIONS:
                    observation = self.backend.refresh_observation()
                after_image = self._capture_primary_image(observation)
                event["cosmos_query_boundary_step"] = policy_step
                event["trigger_distance_m"] = proximity_distance
                event["mean_absolute_raw_pixel_delta"] = _mean_absolute_pixel_delta(
                    before_image, after_image
                )
                self.events.append(event)
                info = dict(info or {})
                info["libero_max_event"] = event

        if self.runtime.applied and original_done:
            self.original_goal_completed_after_event = True

        if policy_step > 0:
            import numpy as np

            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            eef_position = None
            if isinstance(observation, dict) and "robot0_eef_pos" in observation:
                eef_position = np.asarray(
                    observation["robot0_eef_pos"], dtype=np.float64
                ).tolist()
            target_position = None
            trigger = self.scenario["trigger"]
            if trigger["type"] == "on_proximity":
                try:
                    target_position = self.backend.entity_position(trigger["value"])
                except (AttributeError, RuntimeError, ValueError):
                    target_position = None
            self.executed_actions.append(
                {
                    "policy_step": policy_step,
                    "action": action_array.tolist(),
                    "eef_position_m": eef_position,
                    "trigger_entity_position_m": target_position,
                    "original_goal_done": original_done,
                }
            )

        effective_done = original_done
        if self.arm == "intervention" and self.runtime.applied:
            operation = self.scenario["change"]["operation"]
            if operation == "replace_instruction":
                effective_done = self.backend.goal_satisfied(
                    self.scenario["change"].get("alternate_goal")
                )
            elif operation == "cancel_instruction":
                effective_done = False
        return observation, reward, effective_done, info

    def record_policy_query(
        self,
        actions: Any,
        instruction: Optional[str] = None,
        source: str = "model",
    ) -> Dict[str, Any]:
        import numpy as np

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or not np.all(np.isfinite(action_array)):
            raise CosmosIntegrationError(
                "policy action chunk must be a finite rank-2 array"
            )
        policy_step = max(0, self.total_env_steps - self.warmup_steps)
        query = {
            "query_index": len(self.policy_queries),
            "policy_step": policy_step,
            "action_chunk_shape": list(action_array.shape),
            "action_chunk_sha256": hashlib.sha256(
                action_array.tobytes()
            ).hexdigest(),
            "actions": action_array.tolist(),
            "instruction": instruction or self.runtime.current_instruction,
            "source": source,
        }
        self.policy_queries.append(query)
        return query

    @staticmethod
    def _position_delta(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
        import numpy as np

        positions = [row[field] for row in rows if row.get(field) is not None]
        if len(positions) < 2:
            return None
        return float(
            np.linalg.norm(
                np.asarray(positions[-1], dtype=np.float64)
                - np.asarray(positions[0], dtype=np.float64)
            )
        )

    @staticmethod
    def _path_length(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
        import numpy as np

        positions = [
            np.asarray(row[field], dtype=np.float64)
            for row in rows
            if row.get(field) is not None
        ]
        if len(positions) < 2:
            return None
        return float(
            sum(np.linalg.norm(right - left) for left, right in zip(positions, positions[1:]))
        )

    def _intent_response(self) -> Dict[str, Any]:
        operation = self.scenario["change"]["operation"]
        if self.arm != "intervention" or not self.runtime.applied:
            return {"measured": False, "operation": operation}
        event_step = self.events[0]["cosmos_query_boundary_step"]
        response_steps = sorted(
            query["policy_step"]
            for query in self.policy_queries
            if query["policy_step"] >= event_step
            and query.get("instruction") == self.runtime.current_instruction
        )
        response_step = response_steps[0] if response_steps else None
        if operation == "replace_instruction":
            goal_satisfied = self.backend.goal_satisfied(
                self.scenario["change"].get("alternate_goal")
            )
            return {
                "measured": True,
                "operation": operation,
                "response_query_step": response_step,
                "alternate_goal_satisfied": goal_satisfied,
                "correct": bool(goal_satisfied and response_step is not None),
            }

        stop_window_steps = int(self.scenario["change"].get("stop_window_steps", 10))
        eef_threshold = float(
            self.scenario["change"].get("eef_stop_threshold_m", 0.02)
        )
        path_threshold = float(
            self.scenario["change"].get("eef_path_threshold_m", 0.04)
        )
        target_threshold = float(
            self.scenario["change"].get("target_stop_threshold_m", 0.01)
        )
        window = [] if response_step is None else [
            row for row in self.executed_actions if row["policy_step"] > response_step
        ][:stop_window_steps]
        eef_delta = self._position_delta(window, "eef_position_m")
        eef_path = self._path_length(window, "eef_position_m")
        target_delta = self._position_delta(window, "trigger_entity_position_m")
        enough = len(window) == stop_window_steps
        safe_stop = bool(
            enough
            and eef_delta is not None
            and eef_path is not None
            and eef_delta <= eef_threshold
            and eef_path <= path_threshold
            and (target_delta is None or target_delta <= target_threshold)
            and not self.original_goal_completed_after_event
        )
        return {
            "measured": True,
            "operation": operation,
            "response_query_step": response_step,
            "window_steps": len(window),
            "required_window_steps": stop_window_steps,
            "eef_net_displacement_m": eef_delta,
            "eef_path_length_m": eef_path,
            "target_net_displacement_m": target_delta,
            "original_goal_completed_after_event": self.original_goal_completed_after_event,
            "thresholds": {
                "eef_net_displacement_m": eef_threshold,
                "eef_path_length_m": path_threshold,
                "target_net_displacement_m": target_threshold,
            },
            "safe_stop": safe_stop,
            "correct": safe_stop,
        }

    def record_outcome(self, success: bool) -> Dict[str, Any]:
        if self.query_interval is None or self.max_policy_steps is None:
            raise CosmosIntegrationError("configure_episode() was not called")
        if self.init_state_sha256 is None:
            raise CosmosIntegrationError("initial state was not recorded")
        response = None
        response_success = bool(success)
        if self.scenario["change"]["operation"] in INTENT_OPERATIONS:
            response = self._intent_response()
            if self.arm == "intervention" and response.get("measured"):
                response_success = bool(response.get("correct"))
        row = {
            "arm": self.arm,
            "scenario_id": self.scenario["scenario_id"],
            "scenario_seed": self.scenario["seed"],
            "task_suite_name": self.task_suite_name,
            "original_task_index": self.original_task_index,
            "init_state_index": self.init_state_index,
            "task_description": self.task_description,
            "episode_index": self.episode_index,
            "policy_seed": self.policy_seed,
            "init_state_sha256": self.init_state_sha256,
            "query_interval": self.query_interval,
            "max_policy_steps": self.max_policy_steps,
            "total_env_steps": self.total_env_steps,
            "policy_steps": max(0, self.total_env_steps - self.warmup_steps),
            "success": response_success,
            "raw_episode_success": bool(success),
            "intervention_event_count": len(self.events),
            "intervention_events": self.events,
            "setup_event_count": len(self.setup_events),
            "setup_events": self.setup_events,
            "trigger_observation": self.trigger_observation,
            "policy_query_count": len(self.policy_queries),
            "policy_queries": self.policy_queries,
            "executed_actions": self.executed_actions,
            "final_instruction": self.runtime.current_instruction,
            "response_diagnostics": response,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row


def install_cosmos_hooks(
    run_libero_eval: Any,
    scenario: Dict[str, Any],
    arm: str,
    trace_path: Path,
    original_task_index: int,
    init_state_index: int = 0,
    control_trace_path: Optional[Path] = None,
) -> None:
    """Patch the already-imported Cosmos evaluator without editing upstream."""

    original_get_libero_env = run_libero_eval.get_libero_env
    original_run_episode = run_libero_eval.run_episode
    original_load_initial_states = run_libero_eval.load_initial_states
    original_get_action = run_libero_eval.get_action
    active_env: Dict[str, Optional[CosmosInterventionEnv]] = {"value": None}
    control_queries: Dict[int, Any] = {}
    if control_trace_path is not None:
        rows = [
            json.loads(line)
            for line in Path(control_trace_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            raise CosmosIntegrationError(
                "control trace must contain exactly one episode"
            )
        control_queries = {
            query["policy_step"]: query["actions"]
            for query in rows[0].get("policy_queries", [])
        }

    def load_selected_initial_state(cfg: Any, task_suite: Any, task_id: int, log_file=None):
        initial_states, custom_states = original_load_initial_states(
            cfg, task_suite, task_id, log_file
        )
        if custom_states is not None:
            raise CosmosIntegrationError(
                "LIBERO-MAX init_state_index currently requires DEFAULT initial states"
            )
        if not 0 <= init_state_index < len(initial_states):
            raise CosmosIntegrationError(
                "init_state_index %d outside [0, %d)"
                % (init_state_index, len(initial_states))
            )
        return [initial_states[init_state_index]], None

    def get_libero_env_with_intervention(*args: Any, **kwargs: Any):
        env, task_description = original_get_libero_env(*args, **kwargs)
        return (
            CosmosInterventionEnv(
                env=env,
                task_description=task_description,
                scenario=scenario,
                arm=arm,
                trace_path=trace_path,
                original_task_index=original_task_index,
                init_state_index=init_state_index,
            ),
            task_description,
        )

    def run_episode_with_trace(cfg: Any, env: Any, *args: Any, **kwargs: Any):
        if not isinstance(env, CosmosInterventionEnv):
            raise CosmosIntegrationError("Cosmos environment hook was not installed")
        max_steps = run_libero_eval.TASK_MAX_STEPS[cfg.task_suite_name]
        env.configure_episode(
            task_suite_name=cfg.task_suite_name,
            policy_seed=cfg.seed,
            query_interval=cfg.num_open_loop_steps,
            max_policy_steps=max_steps,
        )
        active_env["value"] = env
        try:
            result = original_run_episode(cfg, env, *args, **kwargs)
        finally:
            active_env["value"] = None
        env.record_outcome(result[0])
        return result

    def get_action_with_trace(*args: Any, **kwargs: Any):
        env = active_env["value"]
        instruction = None
        if env is not None:
            instruction = env.runtime.current_instruction
            if len(args) >= 5:
                mutable_args = list(args)
                mutable_args[4] = instruction
                args = tuple(mutable_args)
            elif "task_label_or_embedding" in kwargs:
                kwargs = dict(kwargs)
                kwargs["task_label_or_embedding"] = instruction
        result = original_get_action(*args, **kwargs)
        source = "model"
        if (
            env is not None
            and env.arm == "intervention"
            and not env.runtime.applied
            and control_queries
        ):
            import numpy as np

            policy_step = max(0, env.total_env_steps - env.warmup_steps)
            if policy_step not in control_queries:
                raise CosmosIntegrationError(
                    "control trace is missing pre-event query step %d" % policy_step
                )
            result = dict(result)
            result["actions"] = [
                np.asarray(action, dtype=np.float32)
                for action in control_queries[policy_step]
            ]
            source = "control_replay"
        if env is not None:
            env.record_policy_query(
                result["actions"], instruction=instruction, source=source
            )
        return result

    run_libero_eval.get_libero_env = get_libero_env_with_intervention
    run_libero_eval.run_episode = run_episode_with_trace
    run_libero_eval.load_initial_states = load_selected_initial_state
    run_libero_eval.get_action = get_action_with_trace
