import json
import unittest
from pathlib import Path

from libero_max.scenario import CHANGE_FAMILIES, CHANGE_TYPES, TRIGGER_TYPES


ROOT = Path(__file__).resolve().parents[1]


class JsonSchemaSyncTest(unittest.TestCase):
    def test_scenario_schema_enums_match_runtime(self):
        schema = json.loads(
            (ROOT / "schemas/scenario.schema.json").read_text(encoding="utf-8")
        )
        properties = schema["properties"]
        self.assertEqual(set(properties["change_family"]["enum"]), CHANGE_FAMILIES)
        self.assertEqual(set(properties["change_type"]["enum"]), CHANGE_TYPES)
        self.assertEqual(
            set(properties["trigger"]["properties"]["type"]["enum"]),
            TRIGGER_TYPES,
        )

    def test_result_change_types_match_scenario_schema(self):
        scenario = json.loads(
            (ROOT / "schemas/scenario.schema.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (ROOT / "schemas/result.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            result["properties"]["change_type"]["enum"],
            scenario["properties"]["change_type"]["enum"],
        )


if __name__ == "__main__":
    unittest.main()
