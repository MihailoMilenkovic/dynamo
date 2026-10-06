# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A compact reading order around the complete library-produced request diff."""

import json


def render(report: dict) -> str:
    lines = [
        "# Dynamo versus vLLM request-contract assessment",
        "",
        f"Status: **{report['status']}**. Behavioral conformance: **not assessed**.",
        "",
        "Direction: vLLM serve → Dynamo. Added means declared by Dynamo only; "
        "deleted means declared by vLLM only. These are schema differences, not "
        "automatically confirmed runtime incompatibilities.",
        "",
        "## Inputs",
        "",
    ]
    for side, metadata in report["inputs"].items():
        provenance = metadata["provenance"]
        lines.append(
            f"- {side}: `{provenance['server_version']}`, source "
            f"`{provenance['source_revision']}`; image `{metadata['image']}`; "
            f"raw SHA-256 `{metadata['sha256']}`."
        )
    lines.extend(
        [
            "",
            "The preserved acquisition metadata records startup configuration. "
            "Input metadata is provenance, not cryptographic build attestation.",
            "",
            "## Coverage gaps — address before claiming complete coverage",
            "",
        ]
    )
    for gap in report["coverage_gaps"]:
        lines.append(
            f"- **{gap['side']} {gap['endpoint']}** `{gap.get('field', '*')}`: "
            f"{gap['reason']} Location: `{gap['location']}`."
        )
    if not report["coverage_gaps"]:
        lines.append(
            "No known gaps were detected by the configured coverage checks. "
            "This is not a proof of equivalent accepted requests."
        )
    lines.extend(["", "## Intentional exclusions", ""])
    for exclusion in report["exclusions"]:
        lines.append(f"- `{exclusion['field']}`: {exclusion['reason']}")
    lines.extend(
        [
            "",
            "Responses, inference behavior, forwarding, runtime defaults and "
            "streaming behavior are outside this request-schema assessment.",
            "",
            "## Declared request differences",
            "",
            "See [report.json](report.json) for all structured findings and "
            "[oasdiff.json](oasdiff.json) for the unabridged library result. "
            "Structural union/nullability differences can be representation-only; "
            "review their JSON before calling them wire incompatibilities.",
            "",
        ]
    )
    for finding in report["differences"]:
        lines.extend(
            [
                f"### POST {finding['endpoint']}",
                "",
                f"Input schema location (both retained specs): `{finding['location']}`.",
                "",
            ]
        )
        delta = finding["delta"]
        schema = (
            delta.get("content", {})
            .get("modified", {})
            .get("application/json", {})
            .get("schema", {})
        )
        properties = schema.get("properties", {})
        for key, title in (
            ("deleted", "vLLM-only declarations"),
            ("added", "Dynamo-only declarations"),
        ):
            names = properties.get(key, [])
            lines.append(
                f"- {title}: "
                + (", ".join(f"`{name}`" for name in sorted(names)) or "none")
            )
        changed = properties.get("modified", {})
        lines.append(
            "- Changed fields: "
            + (", ".join(f"`{name}`" for name in sorted(changed)) or "none")
        )
        if schema.get("required"):
            lines.append(
                f"- Required-field changes: `{json.dumps(schema['required'], sort_keys=True)}`"
            )
        lines.extend(
            [
                "",
                "<details>",
                "<summary>Complete request-body delta</summary>",
                "",
                "```json",
                json.dumps(delta, indent=2, sort_keys=True),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    if not report["differences"]:
        lines.append(
            "No declared request-body differences reported. Coverage gaps still apply."
        )
    lines.extend(
        [
            "",
            "## Next actions",
            "",
            "1. Resolve or explicitly track coverage gaps; do not approve them as compatibility.",
            "2. Review each declared difference as an intended divergence, a schema-export defect, "
            "or a protocol change needing implementation.",
            "3. Reacquire and reassess after fixes or dependency/framework version bumps.",
            "4. Run behavioral conformance separately; this report does not satisfy that gate.",
            "",
        ]
    )
    return "\n".join(lines)
