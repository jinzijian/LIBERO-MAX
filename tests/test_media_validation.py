import unittest

import numpy as np

from libero_max.media_validation import validate_render


class MediaValidationTest(unittest.TestCase):
    def test_rejects_random_egl_buffer_except_for_sensor_noise(self):
        normal = np.zeros((64, 64, 3), dtype=np.uint8)
        random_pixels = np.random.default_rng(0).integers(
            0, 256, size=(64, 64, 3), dtype=np.uint8
        )
        with self.assertRaisesRegex(RuntimeError, "EGL render corruption"):
            validate_render("illumination_switch", normal, random_pixels)
        validate_render("sensor_noise_onset", normal, random_pixels)

    def test_rejects_striped_egl_buffer(self):
        normal = np.zeros((64, 64, 3), dtype=np.uint8)
        striped = normal.copy()
        striped[:, 1::2] = 255
        with self.assertRaisesRegex(RuntimeError, "EGL render corruption"):
            validate_render("illumination_switch", normal, striped)


if __name__ == "__main__":
    unittest.main()
