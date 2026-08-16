import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/render_benchmark_media.py"


class RenderMediaSelectionTest(unittest.TestCase):
    def test_selection_covers_distinct_change_types(self):
        # Import only after faking simulator-owned modules is intentionally
        # avoided; selection is source-audited here and exercised remotely in
        # the real MuJoCo environment.
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("def select_representative_cases", source)
        self.assertIn('"illumination_switch"', source)
        self.assertIn('"obstacle_insertion"', source)
        self.assertIn("README_CASE_OVERRIDES", source)
        self.assertIn(
            "pro-task-10-t01-i02-illumination_switch-d1-p195", source
        )

    def test_locked_media_resolution_avoids_corrupt_egl_buffers(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('default=320', source)
        self.assertIn('validated 320 px limit', source)
        self.assertIn('"render_resolution": args.resolution', source)
        self.assertIn('"neighbor_delta_before"', source)
        self.assertIn('"neighbor_delta_after"', source)


if __name__ == "__main__":
    unittest.main()
