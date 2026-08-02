import unittest

from libero_max.libero_backend import LiberoMujocoBackend


class FakeSim:
    pass


class FakeBaseEnv:
    def __init__(self):
        self.sim = FakeSim()


class FakeWrapper:
    def __init__(self):
        self.env = FakeBaseEnv()


class LiberoBackendTest(unittest.TestCase):
    def test_sim_is_resolved_lazily_after_hard_reset(self):
        wrapper = FakeWrapper()
        backend = LiberoMujocoBackend(wrapper)
        original = backend.sim
        replacement = FakeSim()
        wrapper.env.sim = replacement
        self.assertIsNot(original, replacement)
        self.assertIs(backend.sim, replacement)


if __name__ == "__main__":
    unittest.main()
