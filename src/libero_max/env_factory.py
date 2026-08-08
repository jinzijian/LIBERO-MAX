"""Safe construction helpers for LIBERO evaluation environments."""

from typing import Any, Callable, Dict, Tuple


PLACEMENT_FAILURE = "Cannot place all objects"


class RenderStabilityError(RuntimeError):
    """Raised when a new off-screen context returns unstable camera buffers."""


def install_stable_camera_render(
    env: Any, maximum_attempts: int = 6, maximum_neighbor_delta: float = 40.0
) -> Dict[str, Any]:
    """Wrap ``sim.render`` with same-camera readback stabilization.

    MuJoCo 3.2.6 on the locked EGL worker can return a stale buffer from the
    previously rendered camera. Re-rendering the same camera without advancing
    physics converges in two or three reads. The wrapper requires two identical
    smooth RGB buffers, so no stale cross-camera frame reaches a policy.
    """

    import numpy as np

    from .media_validation import mean_neighbor_delta

    if maximum_attempts < 2:
        raise ValueError("stable camera rendering requires at least two attempts")
    inner = getattr(env, "env", env)
    sim = getattr(inner, "sim", None)
    if sim is None or not callable(getattr(sim, "render", None)):
        return {"status": "not_applicable"}
    if getattr(sim, "_libero_max_stable_render_installed", False):
        return sim._libero_max_stable_render_stats

    original_render = sim.render
    stats: Dict[str, Any] = {
        "status": "installed",
        "calls": 0,
        "retries": 0,
        "maximum_attempts_used": 0,
        "maximum_attempts": maximum_attempts,
        "maximum_neighbor_delta": maximum_neighbor_delta,
    }

    def stable_render(*args: Any, **kwargs: Any) -> Any:
        previous_rgb = None
        stats["calls"] += 1
        for attempt in range(1, maximum_attempts + 1):
            result = original_render(*args, **kwargs)
            rgb = result[0] if isinstance(result, tuple) else result
            rgb = np.ascontiguousarray(rgb)
            smooth = (
                rgb.ndim == 3
                and rgb.shape[-1] in {3, 4}
                and mean_neighbor_delta(rgb[..., :3]) <= maximum_neighbor_delta
            )
            if (
                previous_rgb is not None
                and smooth
                and np.array_equal(previous_rgb, rgb)
            ):
                stats["retries"] += attempt - 1
                stats["maximum_attempts_used"] = max(
                    stats["maximum_attempts_used"], attempt
                )
                return result
            previous_rgb = rgb.copy()
        stats["retries"] += maximum_attempts - 1
        stats["maximum_attempts_used"] = maximum_attempts
        stats["failures"] = int(stats.get("failures", 0)) + 1
        raise RenderStabilityError(
            "camera render did not stabilize after %d same-state reads"
            % maximum_attempts
        )

    sim.render = stable_render
    sim._libero_max_stable_render_installed = True
    sim._libero_max_stable_render_stats = stats
    return stats


def _render_snapshot(env: Any) -> Dict[str, Any]:
    """Capture copied raw camera arrays without advancing simulator physics."""

    import numpy as np

    inner = getattr(env, "env", env)
    if not hasattr(inner, "_get_observations"):
        return {}
    if hasattr(inner, "sim"):
        inner.sim.forward()
    if hasattr(inner, "_post_process"):
        inner._post_process()
    if hasattr(inner, "_update_observables"):
        inner._update_observables(force=True)
    observation = inner._get_observations()
    return {
        key: np.ascontiguousarray(value).copy()
        for key, value in observation.items()
        if key.endswith("_image")
        and isinstance(value, np.ndarray)
        and value.ndim == 3
    }


def prime_offscreen_renderer(
    env: Any, attempts: int = 6, maximum_neighbor_delta: float = 40.0
) -> Dict[str, Any]:
    """Require two identical renders before a context reaches the policy.

    The locked multi-GPU EGL stack can occasionally expose an uninitialized
    camera buffer in a newly created context. A valid static MuJoCo state is
    pixel-identical across repeated renders, including the raw substrate used
    by LIBERO-PRO. Contexts that do not stabilize are discarded and rebuilt by
    :func:`create_libero_env_with_retry`.
    """

    import numpy as np

    from .media_validation import mean_neighbor_delta

    if attempts < 2:
        raise ValueError("render priming requires at least two attempts")
    previous = None
    image_keys = []
    for render_attempt in range(1, attempts + 1):
        current = _render_snapshot(env)
        if not current:
            return {"status": "not_applicable", "render_attempts": render_attempt}
        image_keys = sorted(current)
        neighbor_deltas = {
            key: mean_neighbor_delta(current[key]) for key in image_keys
        }
        smooth = max(neighbor_deltas.values()) <= maximum_neighbor_delta
        stable = previous is not None and image_keys == sorted(previous) and all(
            np.array_equal(previous[key], current[key]) for key in image_keys
        )
        if stable and smooth:
            return {
                "status": "passed",
                "render_attempts": render_attempt,
                "image_keys": image_keys,
                "neighbor_deltas": neighbor_deltas,
                "maximum_neighbor_delta": maximum_neighbor_delta,
            }
        previous = current
    raise RenderStabilityError(
        "off-screen camera buffers did not stabilize after %d renders (%s)"
        % (attempts, ", ".join(image_keys))
    )


def create_libero_env_with_retry(
    factory: Callable[[], Tuple[Any, str]],
    *,
    policy_seed: int,
    reseed: Callable[[int], None],
    attempts: int = 10,
) -> Tuple[Any, str]:
    """Construct a LIBERO env despite disposable placement-sampler failures.

    LIBERO samples an initial placement while constructing an environment even
    when the evaluator immediately restores a frozen benchmark state.  Retry
    only that known sampler exhaustion, varying the disposable sampler stream.
    The benchmark policy seed is restored before returning.
    """

    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        reseed(policy_seed + attempt)
        try:
            result = factory()
        except Exception as exc:
            if PLACEMENT_FAILURE not in str(exc) or attempt + 1 == attempts:
                reseed(policy_seed)
                raise
        else:
            env, description = result
            try:
                stable_render = install_stable_camera_render(env)
                render_qa = prime_offscreen_renderer(env)
            except RenderStabilityError:
                if hasattr(env, "close"):
                    env.close()
                if attempt + 1 == attempts:
                    reseed(policy_seed)
                    raise
                continue
            render_qa["stable_camera_render"] = stable_render
            render_qa["environment_attempt"] = attempt + 1
            try:
                setattr(env, "libero_max_render_qa", render_qa)
            except (AttributeError, TypeError):
                pass
            reseed(policy_seed)
            return env, description
    raise AssertionError("unreachable")
