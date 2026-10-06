# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acquire Docker HTTP OpenAPI exports and assess Dynamo/vLLM request contracts."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

from .acquisition.http import acquire
from .assessment.openapi_workflow import DEFAULT_MANIFEST, assess


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "acquire", help="Capture a running pinned container's HTTP export"
    )
    capture.add_argument("--server", choices=("dynamo", "vllm"), required=True)
    capture.add_argument("--container", required=True)
    capture.add_argument(
        "--image",
        required=True,
        help="repository@sha256:digest used to launch the container",
    )
    capture.add_argument(
        "--url", required=True, help="local published HTTP OpenAPI URL"
    )
    capture.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="JSON source/version/dependency/build metadata",
    )
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.set_defaults(run=acquire)
    assessment = commands.add_parser(
        "assess", help="Compose Dynamo and compare its requests to vLLM"
    )
    assessment.add_argument(
        "--dynamo", type=Path, required=True, help="Dynamo capture directory"
    )
    assessment.add_argument(
        "--vllm", type=Path, required=True, help="vLLM capture directory"
    )
    assessment.add_argument(
        "--openai", type=Path, required=True, help="pinned async-openai openapi.yaml"
    )
    assessment.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    assessment.add_argument(
        "--oasdiff", type=Path, required=True, help="pinned oasdiff 1.33.0 executable"
    )
    assessment.add_argument("--output-dir", type=Path, required=True)
    assessment.set_defaults(run=assess)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        subprocess.SubprocessError,
        JsonPatchException,
        JsonPointerException,
        OpenAPIValidationError,
    ) as error:
        print(f"Protocol compatibility tool error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
