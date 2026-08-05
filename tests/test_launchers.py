import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CosmosLauncherTest(unittest.TestCase):
    def test_config_file_is_an_importable_module_path(self) -> None:
        launcher = (ROOT / "scripts/run_cosmos_paired_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "--config_file cosmos_policy/config/config.py",
            launcher,
        )
        self.assertNotIn(
            '--config_file "$COSMOS_POLICY_DIR/cosmos_policy/config/config.py"',
            launcher,
        )

    def test_query_interval_is_forwarded_to_model_and_openpi_client(self) -> None:
        cosmos = (ROOT / "scripts/run_cosmos_paired_smoke.sh").read_text(
            encoding="utf-8"
        )
        openpi = (ROOT / "scripts/run_openpi_paired.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--num_open_loop_steps "$QUERY_INTERVAL"', cosmos)
        self.assertIn('${QUERY_INTERVAL:-5}', openpi)

    def test_preflight_enables_trusted_libero_state_loading_before_imports(self) -> None:
        preflight = (
            ROOT / "scripts/preflight_manifest_interventions.py"
        ).read_text(encoding="utf-8")
        opt_out = (
            'os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")'
        )
        self.assertIn(opt_out, preflight)
        self.assertLess(preflight.index(opt_out), preflight.index("from libero.libero"))
        self.assertLess(
            preflight.index(opt_out),
            preflight.index("from cosmos_policy.experiments.robot.libero"),
        )
        self.assertLess(
            preflight.index('if not hasattr(np, "float_")'),
            preflight.index("from libero.libero"),
        )

    def test_persistent_runner_disables_upstream_rollout_videos(self) -> None:
        runner = (ROOT / "scripts/run_cosmos_persistent_shard.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_libero_eval.save_rollout_video = lambda", runner)
        self.assertIn(
            "run_libero_eval.save_rollout_video_with_future_image_predictions",
            runner,
        )

    def test_persistent_launcher_pins_one_physical_gpu_per_worker(self) -> None:
        launcher = (
            ROOT / "scripts/run_cosmos_persistent_benchmark.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"CUDA_VISIBLE_DEVICES": gpu', launcher)
        self.assertIn('"MUJOCO_EGL_DEVICE_ID": gpu', launcher)
        self.assertIn('"--num-shards"', launcher)

    def test_hard_preflight_uses_plus_overlay_and_physical_egl_ids(self) -> None:
        launcher = (
            ROOT / "scripts/run_max_hard_preflight.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("libero-plus-python-overlay", launcher)
        self.assertIn("LIBERO-plus", launcher)
        self.assertIn("$DEPS_DIR/libero-plus-config}", launcher)
        self.assertNotIn("libero-plus-config/config.yaml", launcher)
        self.assertIn('MUJOCO_EGL_DEVICE_ID="$gpu"', launcher)
        self.assertIn('--num-shards "${#gpu_ids[@]}"', launcher)


if __name__ == "__main__":
    unittest.main()
