"""Pure action-layout and paired-input helpers for the LingBot-VA adapter."""

import hashlib
from typing import Any

import numpy as np


DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


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
