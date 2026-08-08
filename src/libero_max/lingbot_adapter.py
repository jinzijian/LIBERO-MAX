"""Pure action-layout helpers for the LingBot-VA benchmark adapter."""

from typing import Any

import numpy as np


DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


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
