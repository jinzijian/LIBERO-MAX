import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class AuxiliaryPrelaunchTest(unittest.TestCase):
    def test_prelaunch_smokes_and_marks_both_fair_subset_runs(self):
        source = (ROOT / "scripts/run_max8000_auxiliary_prelaunch.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("pro-runtime-compatibility-smoke.json", source)
        self.assertIn('"initial_pose_position_angle"', source)
        self.assertIn('"object_shape"', source)
        self.assertIn('"view_occlusion"', source)
        self.assertIn("run_max_pro_fastwam.sh", source)
        self.assertIn("run_max_pro_lingbot.sh", source)
        self.assertIn("libero_max_pro_model_comparison_800.json", source)
        self.assertIn("RAW_ROLLOUT_FINISHED", source)
        self.assertIn("execution_complete", source)


if __name__ == "__main__":
    unittest.main()
