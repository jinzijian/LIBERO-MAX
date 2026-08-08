import unittest

import numpy as np

from libero_max.env_factory import (
    RenderStabilityError,
    create_libero_env_with_retry,
    prime_offscreen_renderer,
)


class _RenderEnv:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.closed = False

    def _get_observations(self):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return {"agentview_image": frame.copy()}

    def close(self):
        self.closed = True


class EnvFactoryTest(unittest.TestCase):
    def test_primer_requires_two_identical_raw_renders(self):
        corrupt = np.ones((4, 4, 3), dtype=np.uint8)
        stable = np.zeros((4, 4, 3), dtype=np.uint8)
        result = prime_offscreen_renderer(
            _RenderEnv([corrupt, stable, stable]), attempts=3
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["render_attempts"], 3)

    def test_rebuilds_context_that_never_stabilizes(self):
        seeds = []
        bad = _RenderEnv(
            [
                np.full((4, 4, 3), value, dtype=np.uint8)
                for value in range(6)
            ]
        )
        good_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        good = _RenderEnv([good_frame, good_frame])
        environments = iter([bad, good])

        env, _ = create_libero_env_with_retry(
            lambda: (next(environments), "task"),
            policy_seed=195,
            reseed=seeds.append,
            attempts=2,
        )

        self.assertTrue(bad.closed)
        self.assertIs(env, good)
        self.assertEqual(env.libero_max_render_qa["environment_attempt"], 2)
        self.assertEqual(seeds, [195, 196, 195])

    def test_primer_rejects_permanently_unstable_context(self):
        env = _RenderEnv(
            [
                np.full((4, 4, 3), value, dtype=np.uint8)
                for value in range(6)
            ]
        )
        with self.assertRaisesRegex(RenderStabilityError, "did not stabilize"):
            prime_offscreen_renderer(env)

    def test_primer_rejects_stable_but_random_buffer(self):
        random_frame = np.random.default_rng(0).integers(
            0, 256, size=(64, 64, 3), dtype=np.uint8
        )
        env = _RenderEnv([random_frame, random_frame])
        with self.assertRaisesRegex(RenderStabilityError, "did not stabilize"):
            prime_offscreen_renderer(env, attempts=2)

    def test_retries_only_known_placement_failure_and_restores_seed(self):
        seeds = []
        calls = []

        def factory():
            calls.append(len(calls))
            if len(calls) < 3:
                raise RuntimeError("Cannot place all objects ):")
            return object(), "task"

        env, description = create_libero_env_with_retry(
            factory, policy_seed=195, reseed=seeds.append
        )
        self.assertIsNotNone(env)
        self.assertEqual(description, "task")
        self.assertEqual(seeds, [195, 196, 197, 195])

    def test_does_not_mask_unrelated_failure(self):
        seeds = []

        def factory():
            raise RuntimeError("unexpected backend failure")

        with self.assertRaisesRegex(RuntimeError, "unexpected backend failure"):
            create_libero_env_with_retry(
                factory, policy_seed=7, reseed=seeds.append
            )
        self.assertEqual(seeds, [7, 7])

    def test_rejects_invalid_attempt_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            create_libero_env_with_retry(
                lambda: (object(), "task"),
                policy_seed=0,
                reseed=lambda seed: None,
                attempts=0,
            )


if __name__ == "__main__":
    unittest.main()
