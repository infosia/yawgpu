"""Regression tests for the public Dawn GPU identity table generator."""

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "gen_gpu_info", Path(__file__).with_name("gen_gpu_info.py")
)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class GeneratorTests(unittest.TestCase):
    def test_internal_device_set_is_excluded(self):
        data = {
            "_comment": [],
            "vendors": {
                "Test GPU": {
                    "id": "0x1234",
                    "devices": [
                        {"architecture": {"Public GPU": ["0x1001"]}},
                        {
                            "internal": True,
                            "architecture": {"Private GPU": ["0x2002"]},
                        },
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gpu_info.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(GENERATOR, "SOURCE", source):
                generated = GENERATOR.generate()

        rows = re.findall(
            r'\(0x([0-9a-f]+), 0x([0-9a-f]+), "([^"]+)", &\[([^\]]*)\]\)',
            generated,
        )

        def architecture_name(device_id):
            for vendor, mask, name, devices in rows:
                ids = [int(value.strip(), 0) for value in devices.split(",") if value.strip()]
                if int(vendor, 16) == 0x1234 and device_id & int(mask, 16) in ids:
                    return name
            return ""

        self.assertIn('(0x1234, "test-gpu")', generated)
        self.assertEqual(architecture_name(0x1001), "public-gpu")
        self.assertEqual(architecture_name(0x2002), "")
        self.assertNotIn("private-gpu", generated)


if __name__ == "__main__":
    unittest.main()
