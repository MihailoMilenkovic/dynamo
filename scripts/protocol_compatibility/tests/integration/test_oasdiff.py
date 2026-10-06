# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual pinned comparator checks. CI must set OASDIFF_BIN; no mock engine."""

import copy
import os
import tempfile
import unittest
from pathlib import Path

from scripts.protocol_compatibility.acquisition.http import write_json
from scripts.protocol_compatibility.assessment.openapi import (
    compare,
    request_differences,
    request_document,
)
from scripts.protocol_compatibility.tests.assessment.test_openapi import document


class ComparatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        configured = os.environ.get("OASDIFF_BIN")
        if not configured:
            self.skipTest(
                "Set OASDIFF_BIN to validate the real pinned comparator (required in CI)"
            )
        self.binary = Path(configured).resolve()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def differences(self, native, dynamo):
        paths = []
        for name, raw in (("native", native), ("dynamo", dynamo)):
            scoped, gaps = request_document(raw)
            self.assertFalse(gaps)
            path = self.root / f"{name}.json"
            write_json(path, scoped)
            paths.append(path)
        result, _ = compare(self.binary, *paths, flatten=True)
        return request_differences(result)

    def test_response_and_unreachable_components_are_outside_scope(self):
        native = document()
        dynamo = copy.deepcopy(native)
        dynamo["components"]["schemas"]["Response"] = {"type": "string"}
        dynamo["components"]["schemas"]["Unused"] = {"type": "number"}
        self.assertEqual(self.differences(native, dynamo), [])

    def test_simple_allof_representation_does_not_invent_a_difference(self):
        native = document()
        # Remove recursion here: this test is specifically the normal flattened
        # object-export case, not the library's handling of recursive allOf.
        del native["components"]["schemas"]["Request"]["properties"]["recursive"]
        dynamo = copy.deepcopy(native)
        dynamo["components"]["schemas"]["Request"] = {
            "allOf": [dynamo["components"]["schemas"]["Request"]]
        }
        self.assertEqual(self.differences(native, dynamo), [])

    def test_changes_behind_nested_references_and_contract_constraints_are_detected(
        self,
    ):
        pairs = [
            (True, False),
            ({"type": "string", "const": "a"}, {"type": "string", "const": "b"}),
            ({"type": "string", "format": "email"}, {"type": "string"}),
            (
                {"type": "object", "dependentRequired": {"x": ["y"]}},
                {"type": "object"},
            ),
            (
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
                {"type": "array", "items": {"type": "string"}, "minItems": 2},
            ),
            (
                {"type": "number", "exclusiveMinimum": 0},
                {"type": "number", "exclusiveMinimum": 1},
            ),
            (
                {"type": "object", "propertyNames": {"pattern": "^a"}},
                {"type": "object", "propertyNames": {"pattern": "^b"}},
            ),
            ({"not": {"type": "string"}}, {"not": {"type": "number"}}),
            ({"type": "string"}, {"type": "integer"}),
            ({"type": "integer", "minimum": 0}, {"type": "integer", "minimum": 1}),
            ({"type": "string", "enum": ["a", "b"]}, {"type": "string", "enum": ["a"]}),
            ({"type": ["string", "null"]}, {"type": "string"}),
            (
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "integer"}},
            ),
            (
                {"type": "object", "additionalProperties": True},
                {"type": "object", "additionalProperties": False},
            ),
            (
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            ),
        ]
        for before, after in pairs:
            with self.subTest(before=before, after=after):
                native = document()
                native["components"]["schemas"]["Request"]["properties"]["nested"] = {
                    "$ref": "#/components/schemas/Alias"
                }
                native["components"]["schemas"]["Alias"] = {
                    "$ref": "#/components/schemas/Leaf"
                }
                native["components"]["schemas"]["Leaf"] = before
                dynamo = copy.deepcopy(native)
                dynamo["components"]["schemas"]["Leaf"] = after
                self.assertEqual(len(self.differences(native, dynamo)), 2)


if __name__ == "__main__":
    unittest.main()
