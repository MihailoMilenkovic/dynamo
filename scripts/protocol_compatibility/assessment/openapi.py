# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-only projection and coverage guards around the pinned oasdiff engine.

This module selects schemas and reports unsupported constructs. It does not
implement a source parser, schema merger, or semantic compatibility algorithm.
"""

import copy
import json
import subprocess
from pathlib import Path

import jsonpointer
from openapi_spec_validator import validate

ENDPOINTS = ("/v1/chat/completions", "/v1/completions")
OASDIFF_VERSION = "1.33.0"
SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "dependentSchemas")
SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
SCHEMA_SINGLE = (
    "items",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "contentSchema",
)
# These either are unsupported by oasdiff or cannot safely pass through its
# allOf merger. Retain their originals and disable flattening, never drop them.
UNSUPPORTED = {
    "$dynamicRef",
    "$dynamicAnchor",
    "$anchor",
    "$id",
    "$schema",
    "$defs",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "patternProperties",
    "prefixItems",
    "contains",
    "contentSchema",
}
KNOWN = (
    UNSUPPORTED
    | set(SCHEMA_MAPS + SCHEMA_LISTS + SCHEMA_SINGLE)
    | {
        "$ref",
        "$comment",
        "type",
        "enum",
        "const",
        "required",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "dependentRequired",
        "minContains",
        "maxContains",
        "contentEncoding",
        "contentMediaType",
        "description",
        "title",
        "default",
        "example",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "discriminator",
        "externalDocs",
        "xml",
    }
)


def pointer(*parts: str) -> str:
    return "#" + jsonpointer.JsonPointer.from_parts(parts).path


def request_pointer(endpoint: str) -> str:
    return pointer(
        "paths",
        endpoint,
        "post",
        "requestBody",
        "content",
        "application/json",
        "schema",
    )


def schema_nodes(document: dict, schema: dict | bool, location: str, seen=None):
    """Walk schema positions and local component refs; never fetch references.

    Locations refer to retained input documents, including referenced components.
    Literal '$ref' keys in defaults/examples or property names remain data.
    """
    seen = set() if seen is None else seen
    if location in seen:
        return
    seen.add(location)
    yield location, schema
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/components/schemas/"):
            raise ValueError(f"Unsupported request reference at {location}: {ref}")
        yield from schema_nodes(
            document, jsonpointer.resolve_pointer(document, ref[1:]), ref, seen
        )
    for reference in schema.get("discriminator", {}).get("mapping", {}).values():
        if not reference.startswith("#/components/schemas/"):
            raise ValueError(
                f"Unsupported discriminator reference at {location}: {reference}"
            )
        yield from schema_nodes(
            document,
            jsonpointer.resolve_pointer(document, reference[1:]),
            reference,
            seen,
        )
    for key in SCHEMA_MAPS:
        for name, child in schema.get(key, {}).items():
            yield from schema_nodes(
                document, child, location + pointer(key, name)[1:], seen
            )
    for key in SCHEMA_LISTS:
        for index, child in enumerate(schema.get(key, [])):
            yield from schema_nodes(
                document, child, location + pointer(key, str(index))[1:], seen
            )
    for key in SCHEMA_SINGLE:
        if key in schema:
            yield from schema_nodes(
                document, schema[key], location + pointer(key)[1:], seen
            )


def request_document(document: dict) -> tuple[dict, list[dict]]:
    """Keep both POST JSON request contracts and their schema reference closure."""
    if document.get("openapi") not in {"3.1.0", "3.1.1"}:
        raise ValueError("This workflow requires OpenAPI 3.1 server exports")
    if (
        document.get(
            "jsonSchemaDialect", "https://spec.openapis.org/oas/3.1/dialect/base"
        )
        != "https://spec.openapis.org/oas/3.1/dialect/base"
    ):
        raise ValueError(
            "Custom OpenAPI schema dialects need explicit comparison support"
        )
    result = {
        "openapi": document["openapi"],
        "info": {"title": "Request contracts", "version": "1"},
        "paths": {},
        "components": {"schemas": {}},
    }
    gaps = []
    for endpoint in ENDPOINTS:
        body = copy.deepcopy(document["paths"][endpoint]["post"]["requestBody"])
        if set(body.get("content", {})) != {"application/json"}:
            raise ValueError(f"Expected one JSON request media type for {endpoint}")
        schema = body["content"]["application/json"]["schema"]
        for location, node in schema_nodes(document, schema, request_pointer(endpoint)):
            if location.startswith("#/components/schemas/"):
                name = jsonpointer.JsonPointer(location[1:]).parts[2]
                result["components"]["schemas"][name] = copy.deepcopy(
                    document["components"]["schemas"][name]
                )
            if isinstance(node, dict):
                if "x-dynamo-schema-import" in node:
                    raise ValueError(f"Unresolved dependency schema at {location}")
                unsupported = UNSUPPORTED.intersection(node) | {
                    key for key in node if key not in KNOWN and not key.startswith("x-")
                }
                for keyword in sorted(unsupported):
                    gaps.append(
                        {
                            "endpoint": endpoint,
                            "location": location,
                            "reason": f"Comparison/flattening support is incomplete for {keyword}",
                        }
                    )
        # No responses, parameters, security, operation IDs or unrelated endpoints
        # are compared. A fixed empty response is required by OpenAPI itself.
        result["paths"][endpoint] = {
            "post": {
                "requestBody": body,
                "responses": {"200": {"description": "Outside assessment scope"}},
            }
        }
    validate(result)
    return result, gaps


def compare(
    binary: Path, native: Path, dynamo: Path, *, flatten: bool
) -> tuple[dict, list[str]]:
    version = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if version.stdout.strip() != f"oasdiff version {OASDIFF_VERSION}":
        raise ValueError(
            f"Expected oasdiff {OASDIFF_VERSION}, got {version.stdout.strip()}"
        )
    command = [
        str(binary),
        "diff",
        "--allow-external-refs=false",
        "--auto-upgrade",
        "--exclude-elements",
        "description,examples,extensions,summary,title",
        "--format",
        "json",
    ]
    if flatten:
        command.append("--flatten-allof")
    command.extend([str(native), str(dynamo)])
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    return json.loads(result.stdout or "{}"), command


def request_differences(diff: dict) -> list[dict]:
    """Expose complete request-body deltas, not response/component rename noise."""
    findings = []
    for endpoint, change in sorted(diff.get("paths", {}).get("modified", {}).items()):
        operation = change.get("operations", {}).get("modified", {}).get("POST", {})
        if operation.get("requestBody"):
            findings.append(
                {
                    "endpoint": endpoint,
                    "location": request_pointer(endpoint),
                    "delta": operation["requestBody"],
                }
            )
    if diff.get("paths", {}).get("added") or diff.get("paths", {}).get("deleted"):
        raise ValueError("Scoped endpoint disappeared during comparison")
    return findings
