import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ModelLauncherIsolationTest(unittest.TestCase):
    def test_model_venv_precedes_shared_libero_site_packages(self):
        for name in ("run_max_pro_fastwam.sh", "run_max_pro_lingbot.sh"):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("ACTIVE_SITE_PACKAGES", source)
            active = source.index('export PYTHONPATH="$ACTIVE_SITE_PACKAGES:')
            shared = source.index("$DEPS_DIR/.venv-libero/lib/python3.10/site-packages")
            self.assertLess(active, shared)
            self.assertIn('mujoco.__version__ != "3.2.6"', source)
            self.assertIn('hasattr(mujoco.MjModel, "mesh_scale")', source)

    def test_openpi_pro_launcher_pins_pro_runtime_paths(self):
        source = (ROOT / "scripts" / "run_max_pro_openpi.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("LIBERO-PRO", source)
        self.assertIn("libero-pro-python-overlay", source)
        self.assertIn("robosuite-1.4.0", source)

    def test_openpi_server_prefers_its_own_venv(self):
        source = (ROOT / "scripts" / "run_openpi_persistent_benchmark.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("OPENPI_FALLBACK_PYTHON", source)
        self.assertIn("openpi_runtime_ready", source)
        self.assertIn('"python": sys.executable', source)
        self.assertIn('mujoco.__version__ != "3.2.6"', source)
        self.assertIn('hasattr(mujoco.MjModel, "mesh_scale")', source)
        self.assertIn('"simulator_client": json.loads(sys.argv[7])', source)
        self.assertIn("OPENPI_SITE_PACKAGES", source)
        self.assertIn("OPENPI_SERVER_PYTHONPATH", source)
        self.assertIn('PYTHONPATH="$OPENPI_SERVER_PYTHONPATH"', source)
        declaration = source.index('OPENPI_SERVER_PYTHONPATH="')
        self.assertNotIn(
            "${PYTHONPATH:+:$PYTHONPATH}",
            source[declaration : source.index("CLIENT_PYTHON=", declaration)],
        )
        self.assertIn("os.kill(server_pid, 0)", source)
        self.assertIn("exited before port", source)


if __name__ == "__main__":
    unittest.main()
