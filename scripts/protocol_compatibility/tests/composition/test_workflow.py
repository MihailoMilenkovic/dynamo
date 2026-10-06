# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.protocol_compatibility.acquisition.http import load_capture, write_json
from scripts.protocol_compatibility.assessment.openapi_workflow import assess
from scripts.protocol_compatibility.tests.assessment.test_openapi import document


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for side in ("dynamo", "vllm"):
            directory = self.root / side
            directory.mkdir()
            write_json(directory / "openapi.raw.json", document())
            write_json(
                directory / "acquisition.json",
                {
                    "schema": "protocol-http-acquisition/v1",
                    "server": side,
                    "sha256": hashlib.sha256(
                        (directory / "openapi.raw.json").read_bytes()
                    ).hexdigest(),
                    "image": "fixture@sha256:" + "0" * 64,
                    "provenance": {
                        "source_revision": "a" * 40,
                        "server_version": "fixture",
                        "dependencies": {"fixture": "1"},
                    },
                },
            )
        self.baseline = self.root / "openai.yaml"
        self.baseline.write_text("components: {schemas: {}}\n")
        self.manifest = self.root / "manifest.yaml"
        self.manifest.write_text(
            yaml.safe_dump(
                {
                    "openai": {
                        "sha256": hashlib.sha256(self.baseline.read_bytes()).hexdigest()
                    },
                    "dependencies": {"fixture": "1"},
                    "imports": {},
                    "exclusions": [
                        {"field": "nvext", "reason": "Dynamo-only contents"}
                    ],
                }
            )
        )
        self.binary = self.root / "oasdiff"
        self.binary.write_bytes(b"mock binary, not executable")
        self.args = SimpleNamespace(
            dynamo=self.root / "dynamo",
            vllm=self.root / "vllm",
            openai=self.baseline,
            manifest=self.manifest,
            oasdiff=self.binary,
            output_dir=self.root / "report",
        )

    def test_modified_capture_is_rejected(self):
        (self.root / "dynamo/openapi.raw.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after acquisition"):
            load_capture(self.root / "dynamo", "dynamo")

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_equal_schemas_do_not_hide_known_coverage_gaps(self, compare):
        self.assertEqual(assess(self.args), 1)
        result = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(result["status"], "incomplete_coverage")
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["behavioral_conformance"], "not_assessed")
        self.assertEqual(
            (self.args.output_dir / "dynamo/openapi.raw.json").read_bytes(),
            (self.args.dynamo / "openapi.raw.json").read_bytes(),
        )
        self.assertTrue((self.args.output_dir / "dynamo.composed.json").exists())
        markdown = (self.args.output_dir / "report.md").read_text()
        self.assertIn("Intentional exclusions", markdown)
        self.assertIn("Coverage gaps", markdown)
        with self.assertRaises(FileExistsError):
            assess(self.args)
        self.assertEqual(compare.call_count, 1)

    def test_dependency_pin_mismatch_fails_before_writing(self):
        manifest = yaml.safe_load(self.manifest.read_text())
        manifest["dependencies"] = {"fixture": "2"}
        self.manifest.write_text(yaml.safe_dump(manifest))
        with self.assertRaisesRegex(ValueError, "dependency metadata"):
            assess(self.args)
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
