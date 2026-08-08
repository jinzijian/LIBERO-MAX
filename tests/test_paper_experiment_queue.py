import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_max8000_paper_queue.sh"
STATUS_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_experiment_status.py"


class PaperExperimentQueueTest(unittest.TestCase):
    def test_queue_preserves_evidence_gates_and_deliverables(self):
        source = SCRIPT.read_text(encoding="utf-8")
        status_source = STATUS_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("assert_complete", source)
        self.assertIn("build_infrastructure_repair_manifest.py", source)
        self.assertIn("trigger-unreached", source)
        self.assertIn("response_query_unreached", status_source)
        self.assertIn("benchmark/max8000/libero_max_8000.json", source)
        self.assertIn("build_paper_tables.py", source)
        self.assertIn("build_paper_figures.py", source)
        self.assertIn("build_paper_appendix.py", source)
        self.assertIn("build_experiment_status.py", source)
        self.assertIn("build_paper_analysis.py", source)
        self.assertIn("ensure_rollouts", source)
        self.assertIn("RAW_ROLLOUT_FINISHED", source)
        self.assertIn("tables/ablation", source)
        self.assertIn("tables/tracks", source)
        self.assertIn("build_human_review_queue.py", source)
        self.assertIn("render_benchmark_media.py", source)
        self.assertIn("render_rollout_replay.py", source)
        self.assertIn("select_cosmos_replay_case", source)
        self.assertIn("PAPER_QUEUE_DONE", source)
        self.assertIn("LIBERO-plus", source)
        self.assertIn("LIBERO-PRO", source)
        self.assertIn("OPENPI_SYNC_PID_FILE", source)
        self.assertIn("frozen/intent/cosmos=", source)
        self.assertIn("frozen/ablation/cosmos-notified-q16=", source)


if __name__ == "__main__":
    unittest.main()
