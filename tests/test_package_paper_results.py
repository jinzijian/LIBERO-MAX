import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.package_paper_results import REQUIRED_MEDIA


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "package_paper_results.py"


class PackagePaperResultsTest(unittest.TestCase):
    def test_packages_only_compact_evidence_and_media(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            paper = temp / "paper"
            output = temp / "results"
            media = temp / "assets"
            run = paper / "runs" / "model"
            table = paper / "tables" / "main"
            review = paper / "human_review"
            media_source = paper / "media"
            figure = paper / "figures" / "main"
            appendix = paper / "paper"
            for path in (run, table, review, media_source, figure, appendix):
                path.mkdir(parents=True)
            (paper / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            (run / "benchmark_summary.json").write_text("{}")
            (run / "end_to_end_results.jsonl").write_text("{}\n")
            (run / "paired_results.jsonl").write_text("{}\n")
            (run / "manifest.json").write_text("{}\n")
            (run / "run_config.json").write_text("{}\n")
            (run / "raw_trace.bin").write_bytes(b"large trace")
            (table / "main_results.md").write_text("table")
            (table / "main_results.tex").write_text("\\begin{table}\n")
            (review / "human_review_queue.csv").write_text("case_id\n")
            (figure / "overall_success.png").write_bytes(b"PNG")
            (appendix / "MAX8000_RESULTS.md").write_text("appendix")
            for name in REQUIRED_MEDIA:
                (media_source / name).write_bytes(b"validated-media")
            (media_source / "preview.gif").write_bytes(b"GIF89a")
            (media_source / "preview.trace.jsonl").write_text("large trace\n")

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(paper),
                    str(output),
                    "--media-output-dir",
                    str(media),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output / "runs/model/benchmark_summary.json").exists())
            self.assertTrue((output / "SHA256SUMS").exists())
            self.assertFalse((output / "runs/model/raw_trace.bin").exists())
            self.assertTrue((output / "figures/main/overall_success.png").exists())
            self.assertTrue((output / "tables/main/main_results.tex").exists())
            self.assertTrue((output / "paper/MAX8000_RESULTS.md").exists())
            self.assertTrue((media / "preview.gif").exists())
            self.assertTrue((media / "SHA256SUMS").exists())
            self.assertIn("preview.gif", (media / "SHA256SUMS").read_text())
            self.assertFalse((media / "preview.trace.jsonl").exists())

    def test_refuses_complete_bundle_with_missing_run_provenance(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            paper = temp / "paper"
            run = paper / "runs" / "model"
            run.mkdir(parents=True)
            (paper / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            (run / "benchmark_summary.json").write_text("{}")
            (run / "end_to_end_results.jsonl").write_text("{}\n")
            (run / "paired_results.jsonl").write_text("{}\n")
            (run / "manifest.json").write_text("{}\n")

            completed = subprocess.run(
                ["python3", str(SCRIPT), str(paper), str(temp / "results")],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("required paper artifact is missing", completed.stderr)


if __name__ == "__main__":
    unittest.main()
