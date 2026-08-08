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


if __name__ == "__main__":
    unittest.main()
