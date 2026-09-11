"""Small standard-library checks; numerical model check is benchmark.py --check."""

import ast
import csv
import json
import tempfile
from pathlib import Path
import unittest

from benchmark import FIELDS, percentile, summarize
from plot_results import load_rows


class MeasurementSemantics(unittest.TestCase):
    def test_notebook_sources_match_scripts(self):
        root = Path(__file__).parent
        notebook = json.loads((root / "week2-inference-lab.ipynb").read_text())
        embedded = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertFalse(cell["outputs"])
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets):
                    embedded = ast.literal_eval(node.value)
        self.assertIsNotNone(embedded)
        for name, source in embedded.items():
            self.assertEqual(source, (root / name).read_text(), name)

    def test_output_first_token_and_decode_denominator(self):
        # Explicit synthetic test fixture, never written to delivered results.
        # Five output tokens require four decode calls; 40 ms / 4 = 10 ms.
        row = summarize(3, 7, 5, [12, 10, 11], [40, 44, 36], 2_000_000_000)
        self.assertEqual(row["decode_steps"], 4)
        self.assertEqual(row["prefill_ms_p50"], 11)
        self.assertEqual(row["decode_step_ms_p50"], 10)
        self.assertEqual(row["per_user_tokens_s"], 100)
        self.assertEqual(row["aggregate_output_tokens_s"], 300)
        self.assertEqual(row["peak_allocated_gb"], 2)

    def test_quantile(self):
        self.assertEqual(percentile([30, 10, 20], 0.5), 20)
        self.assertEqual(percentile([30, 10, 20], 0.95), 29)
        self.assertRaises(ValueError, percentile, [], 0.5)

    def test_oom_not_plotted_as_zero(self):
        row = summarize(2, 7, 5, [10] * 3, [40] * 3, 1_000_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
                writer.writerow({"batch_size": 64, "status": "oom"})
            loaded = load_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
