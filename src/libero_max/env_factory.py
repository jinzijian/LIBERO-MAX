"""Safe construction helpers for LIBERO evaluation environments."""

from typing import Any, Callable, Dict, Tuple


PLACEMENT_FAILURE = "Cannot place all objects"


class RenderStabilityError(RuntimeError):
    """Raised when a new off-screen context returns unstable camera buffers."""


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
                render_qa = prime_offscreen_renderer(env)
            except RenderStabilityError:
                if hasattr(env, "close"):
                    env.close()
                if attempt + 1 == attempts:
                    reseed(policy_seed)
                    raise
                continue
            render_qa["environment_attempt"] = attempt + 1
            try:
                setattr(env, "libero_max_render_qa", render_qa)
            except (AttributeError, TypeError):
                pass
            reseed(policy_seed)
            return env, description
    raise AssertionError("unreachable")
