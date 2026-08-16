"""Pure action-layout and paired-input helpers for the LingBot-VA adapter."""

import hashlib
from typing import Any

import numpy as np


DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
MAXIMUM_PAIRED_RENDER_MEAN_ABSOLUTE_DELTA = 0.25


def lingbot_policy_input_digests(images: Any, sim_state: Any) -> dict:
    """Hash the exact policy images and MuJoCo state at one query boundary."""

    if not isinstance(images, dict) or not images:
        raise ValueError("LingBot policy images must be a non-empty mapping")
    image_hashes = {}
    for key, value in sorted(images.items()):
        array = np.ascontiguousarray(value)
        if array.ndim != 3:
            raise ValueError("LingBot policy images must be rank-3 arrays")
        image_hashes[key] = hashlib.sha256(array.tobytes()).hexdigest()
    state = np.ascontiguousarray(np.asarray(sim_state))
    return {
        "policy_image_sha256": image_hashes,
        "sim_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
    }


def compare_lingbot_paired_inputs(
    control_images: Any,
    intervention_images: Any,
    control_digests: dict,
    intervention_digests: dict,
) -> dict:
    """Require exact physics and bounded same-state EGL render variation.

    MuJoCo state must be byte-identical.  The locked EGL rasterizer can vary a
    small number of eye-in-hand pixels even when that state is identical, so
    image equivalence uses the same 0.25 mean-absolute-pixel threshold as the
    benchmark's repeated-render initialization gate.  Exact image hashes and
    the measured per-camera deltas remain in the trace.
    """

    if control_digests.get("sim_state_sha256") != intervention_digests.get(
        "sim_state_sha256"
    ):
        return {
            "status": "failed",
            "reason": "sim_state_sha256_mismatch",
            "sim_state_exact": False,
        }
    if not isinstance(control_images, dict) or not isinstance(
        intervention_images, dict
    ):
        return {"status": "failed", "reason": "images_not_mappings"}
    if set(control_images) != set(intervention_images):
        return {"status": "failed", "reason": "image_keys_mismatch"}

    deltas = {}
    for key in sorted(control_images):
        left = np.asarray(control_images[key])
        right = np.asarray(intervention_images[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            return {
                "status": "failed",
                "reason": "image_layout_mismatch",
                "image_key": key,
            }
        absolute = np.abs(left.astype(np.int16) - right.astype(np.int16))
        deltas[key] = {
            "mean_absolute_delta": float(absolute.mean()),
            "maximum_absolute_delta": int(absolute.max(initial=0)),
            "changed_values": int(np.count_nonzero(absolute)),
            "total_values": int(absolute.size),
        }
    maximum_mean = max(
        (row["mean_absolute_delta"] for row in deltas.values()), default=0.0
    )
    exact_images = control_digests.get(
        "policy_image_sha256"
    ) == intervention_digests.get("policy_image_sha256")
    passed = maximum_mean <= MAXIMUM_PAIRED_RENDER_MEAN_ABSOLUTE_DELTA
    return {
        "status": "passed" if passed else "failed",
        "reason": None if passed else "render_delta_above_threshold",
        "sim_state_exact": True,
        "images_byte_exact": exact_images,
        "maximum_allowed_mean_absolute_delta": (
            MAXIMUM_PAIRED_RENDER_MEAN_ABSOLUTE_DELTA
        ),
        "maximum_observed_mean_absolute_delta": maximum_mean,
        "image_deltas": deltas,
    }


def flatten_lingbot_actions(native: Any, query_index: int) -> np.ndarray:
    action = np.asarray(native, dtype=np.float32)
    if action.shape != (7, 4, 4):
        raise ValueError(
            "LingBot returned action shape %r, expected (7, 4, 4)" % (action.shape,)
        )
    flat = np.transpose(action, (1, 2, 0)).reshape(16, 7).astype(np.float32)
    if query_index == 0:
        # LingBot's first action frame is a conditioning frame. The official
        # client does not execute it. Fill those four physical q16 slots with
        # standard LIBERO no-ops while retaining the native tensor for the
        # model's cache update.
        flat[:4] = DUMMY_ACTION
    return flat
