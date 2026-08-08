import unittest

from libero_max.env_factory import create_libero_env_with_retry


class EnvFactoryTest(unittest.TestCase):
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
