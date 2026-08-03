"""Non-invasive LIBERO-MAX hooks for the upstream Cosmos Policy evaluator."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        if scenario["change"]["operation"] in INTENT_OPERATIONS:
            raise CosmosIntegrationError(
                "Cosmos wrapper currently supports physical changes only"
            )
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
        return observation, reward, done, info

    def record_policy_query(self, actions: Any) -> Dict[str, Any]:
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
        }
        self.policy_queries.append(query)
        return query

    def record_outcome(self, success: bool) -> Dict[str, Any]:
        if self.query_interval is None or self.max_policy_steps is None:
            raise CosmosIntegrationError("configure_episode() was not called")
        if self.init_state_sha256 is None:
            raise CosmosIntegrationError("initial state was not recorded")
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
            "success": bool(success),
            "intervention_event_count": len(self.events),
            "intervention_events": self.events,
            "setup_event_count": len(self.setup_events),
            "setup_events": self.setup_events,
            "trigger_observation": self.trigger_observation,
            "policy_query_count": len(self.policy_queries),
            "policy_queries": self.policy_queries,
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
) -> None:
    """Patch the already-imported Cosmos evaluator without editing upstream."""

    original_get_libero_env = run_libero_eval.get_libero_env
    original_run_episode = run_libero_eval.run_episode
    original_load_initial_states = run_libero_eval.load_initial_states
    original_get_action = run_libero_eval.get_action
    active_env: Dict[str, Optional[CosmosInterventionEnv]] = {"value": None}

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
        result = original_get_action(*args, **kwargs)
        env = active_env["value"]
        if env is not None:
            env.record_policy_query(result["actions"])
        return result

    run_libero_eval.get_libero_env = get_libero_env_with_intervention
    run_libero_eval.run_episode = run_episode_with_trace
    run_libero_eval.load_initial_states = load_selected_initial_state
    run_libero_eval.get_action = get_action_with_trace
