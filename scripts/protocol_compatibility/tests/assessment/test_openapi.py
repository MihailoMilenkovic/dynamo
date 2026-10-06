# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.protocol_compatibility.assessment.openapi import (
    ENDPOINTS,
    compare,
    request_differences,
    request_document,
    request_pointer,
)


def document():
    return {
        "openapi": "3.1.0",
        "info": {"title": "Fixture", "version": "1"},
        "paths": {
            endpoint: {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Request"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Not compared",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Response"}
                                },
                            },
                        }
                    },
                }
            }
            for endpoint in ENDPOINTS
        },
        "components": {
            "schemas": {
                "Request": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "recursive": {"$ref": "#/components/schemas/Request"},
                        "$ref": {"type": "string"},
                    },
                    "required": ["value"],
                    "default": {"value": "x", "$ref": "literal"},
                },
                "Response": {"type": "integer"},
                "Unused": {"type": "boolean"},
            }
        },
    }


class RequestProjectionTests(unittest.TestCase):
    def test_custom_dialect_is_not_silently_replaced_by_default(self):
        raw = document()
        raw["jsonSchemaDialect"] = "https://example.invalid/custom-dialect"
        with self.assertRaisesRegex(ValueError, "Custom OpenAPI schema dialects"):
            request_document(raw)

    def test_only_scoped_requests_and_reference_closure_survive(self):
        raw = document()
        raw["paths"]["/unrelated/{id}"] = {"get": {"responses": {}}}
        before = copy.deepcopy(raw)
        scoped, gaps = request_document(raw)
        self.assertEqual(raw, before)
        self.assertFalse(gaps)
        self.assertEqual(set(scoped["paths"]), set(ENDPOINTS))
        self.assertEqual(set(scoped["components"]["schemas"]), {"Request"})
        self.assertEqual(
            scoped["components"]["schemas"]["Request"]["default"],
            {"value": "x", "$ref": "literal"},
        )
        self.assertNotIn(
            "content", scoped["paths"][ENDPOINTS[0]]["post"]["responses"]["200"]
        )

    def test_external_refs_and_unresolved_slots_fail_visibly(self):
        raw = document()
        raw["components"]["schemas"]["Request"]["properties"]["value"] = {
            "$ref": "https://example.com/schema"
        }
        with self.assertRaisesRegex(ValueError, "Unsupported request reference"):
            request_document(raw)
        raw = document()
        raw["components"]["schemas"]["Request"]["x-dynamo-schema-import"] = {
            "type": "Missing"
        }
        with self.assertRaisesRegex(ValueError, "Unresolved dependency"):
            request_document(raw)

    def test_unhandled_constraints_remain_visible_and_report_gaps(self):
        raw = document()
        raw["components"]["schemas"]["Request"]["if"] = {"required": ["conditional"]}
        raw["components"]["schemas"]["Request"]["then"] = {"required": ["value"]}
        scoped, gaps = request_document(raw)
        self.assertEqual(len(gaps), 4)
        self.assertIn("if", scoped["components"]["schemas"]["Request"])
        self.assertTrue(
            all(gap["location"] == "#/components/schemas/Request" for gap in gaps)
        )

    def test_missing_endpoint_or_media_type_is_not_empty_success(self):
        raw = document()
        del raw["paths"][ENDPOINTS[0]]
        with self.assertRaises(KeyError):
            request_document(raw)
        raw = document()
        raw["paths"][ENDPOINTS[0]]["post"]["requestBody"]["content"]["text/plain"] = {}
        with self.assertRaisesRegex(ValueError, "media type"):
            request_document(raw)

    def test_differences_keep_nested_details_and_ignore_response_noise(self):
        delta = {
            "content": {
                "modified": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "modified": {
                                    "nested": {
                                        "type": {"from": "string", "to": "integer"}
                                    }
                                }
                            },
                        }
                    }
                }
            }
        }
        diff = {
            "components": {"schemas": {"added": ["Renamed"]}},
            "paths": {
                "modified": {
                    ENDPOINTS[0]: {
                        "operations": {"modified": {"POST": {"requestBody": delta}}}
                    },
                    ENDPOINTS[1]: {
                        "operations": {
                            "modified": {"POST": {"responses": {"deleted": ["400"]}}}
                        }
                    },
                }
            },
        }
        self.assertEqual(
            request_differences(diff),
            [
                {
                    "endpoint": ENDPOINTS[0],
                    "location": request_pointer(ENDPOINTS[0]),
                    "delta": delta,
                }
            ],
        )
        self.assertEqual(request_differences({}), [])

    @patch("scripts.protocol_compatibility.assessment.openapi.subprocess.run")
    def test_comparator_pin_network_restriction_and_flatten_guard(self, run):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "oasdiff version 1.33.0\n"),
            subprocess.CompletedProcess([], 0, json.dumps({})),
        ]
        result, command = compare(
            Path("/fixture/oasdiff"), Path("v.json"), Path("d.json"), flatten=False
        )
        self.assertEqual(result, {})
        self.assertIn("--allow-external-refs=false", command)
        self.assertNotIn("--flatten-allof", command)
        self.assertEqual(command[-2:], ["v.json", "d.json"])
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "oasdiff version 9.0.0\n")
        ]
        with self.assertRaisesRegex(ValueError, "Expected oasdiff"):
            compare(
                Path("/fixture/oasdiff"), Path("v.json"), Path("d.json"), flatten=True
            )


if __name__ == "__main__":
    unittest.main()
