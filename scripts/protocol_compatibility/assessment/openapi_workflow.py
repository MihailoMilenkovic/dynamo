# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect independently acquired HTTP exports, composition, and comparison."""

import hashlib
import shutil
from pathlib import Path

import yaml

from ..acquisition.http import load_capture, write_json
from ..composition.openai import compose
from ..reporting.openapi import render
from .openapi import compare, request_differences, request_document

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE / "composition/async-openai-0.41.1.yaml"


def assess(args) -> int:
    dynamo_raw, dynamo_metadata = load_capture(args.dynamo, "dynamo")
    native_raw, native_metadata = load_capture(args.vllm, "vllm")
    manifest_bytes = args.manifest.read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    if dynamo_metadata["provenance"]["dependencies"] != manifest["dependencies"]:
        raise ValueError(
            "Dynamo dependency metadata does not match the reviewed import manifest"
        )
    composed = compose(dynamo_raw, manifest, baseline_bytes=args.openai.read_bytes())
    dynamo, dynamo_gaps = request_document(composed)
    native, native_gaps = request_document(native_raw)
    gaps = [
        {"side": side, **gap}
        for side, items in (("dynamo", dynamo_gaps), ("vllm", native_gaps))
        for gap in items
    ]
    coverage_bytes = (PACKAGE / "assessment/coverage.yaml").read_bytes()
    gaps.extend(
        {"side": "dynamo", **gap} for gap in yaml.safe_load(coverage_bytes)["gaps"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for side, source in (("dynamo", args.dynamo), ("vllm", args.vllm)):
        destination = args.output_dir / side
        destination.mkdir()
        for name in ("openapi.raw.json", "acquisition.json"):
            shutil.copyfile(source / name, destination / name)
    (args.output_dir / "openai.yaml").write_bytes(args.openai.read_bytes())
    (args.output_dir / "composition.yaml").write_bytes(manifest_bytes)
    (args.output_dir / "coverage.yaml").write_bytes(coverage_bytes)
    write_json(args.output_dir / "dynamo.composed.json", composed)
    dynamo_path, native_path = (
        args.output_dir / "dynamo.requests.json",
        args.output_dir / "vllm.requests.json",
    )
    write_json(dynamo_path, dynamo)
    write_json(native_path, native)
    raw_diff, command = compare(
        args.oasdiff, native_path, dynamo_path, flatten=not (dynamo_gaps or native_gaps)
    )
    findings = request_differences(raw_diff)
    report = {
        "schema": "dynamo-vllm-openapi-assessment/v1",
        "status": (
            "incomplete_coverage"
            if gaps
            else ("differences_found" if findings else "no_declared_differences")
        ),
        "behavioral_conformance": "not_assessed",
        "differences": findings,
        "coverage_gaps": gaps,
        "exclusions": manifest["exclusions"],
        "inputs": {"dynamo": dynamo_metadata, "vllm": native_metadata},
        "composition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "coverage_catalog_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "comparator": {
            "command": command,
            "binary_sha256": hashlib.sha256(args.oasdiff.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output_dir / "oasdiff.json", raw_diff)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(render(report))
    print(
        f"{report['status']}: {len(findings)} endpoint request deltas; {len(gaps)} coverage gaps. "
        f"Read {args.output_dir / 'report.md'}"
    )
    return int(bool(findings or gaps))
