"""Validation guards for simulator media artifacts."""

import numpy as np


def mean_neighbor_delta(image: np.ndarray) -> float:
    values = image.astype(np.float32)
    horizontal = np.abs(values[:, 1:] - values[:, :-1]).mean()
    vertical = np.abs(values[1:] - values[:-1]).mean()
    return float((horizontal + vertical) / 2.0)


def validate_render(change_type: str, before: np.ndarray, after: np.ndarray) -> None:
    """Reject near-random EGL buffers while allowing intentional sensor noise."""

    if change_type == "sensor_noise_onset":
        return
    maximum_delta = max(mean_neighbor_delta(before), mean_neighbor_delta(after))
    if maximum_delta > 60.0:
        raise RuntimeError(
            "suspected EGL render corruption: neighbor delta %.2f" % maximum_delta
        )
