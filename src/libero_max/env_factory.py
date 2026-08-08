"""Safe construction helpers for LIBERO evaluation environments."""

from typing import Any, Callable, Tuple


PLACEMENT_FAILURE = "Cannot place all objects"


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
            reseed(policy_seed)
            return result
    raise AssertionError("unreachable")
