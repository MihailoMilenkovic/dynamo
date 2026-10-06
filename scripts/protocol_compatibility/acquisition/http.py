# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture an existing pinned Docker server's HTTP export without managing it."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, stream, code, message, headers, url):
        return None


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def acquire(args) -> int:
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", args.image):
        raise ValueError("Use an immutable repository@sha256 image reference")
    parsed = urlsplit(args.url)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("Acquire from a locally published Docker HTTP port")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The acquisition URL must not contain credentials, query or fragment"
        )
    container = json.loads(
        subprocess.run(
            ["docker", "inspect", args.container],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )[0]
    if not container["State"]["Running"] or container["Config"]["Image"] != args.image:
        raise ValueError(
            "Container must be running with the exact requested digest reference"
        )
    bindings = container["NetworkSettings"]["Ports"]
    if not any(
        binding["HostPort"] == str(parsed.port or 80)
        for values in bindings.values()
        for binding in (values or [])
    ):
        raise ValueError(
            "Acquisition URL port is not published by the selected container"
        )
    with build_opener(ProxyHandler({}), NoRedirect).open(
        args.url, timeout=30
    ) as response:
        if response.url != args.url:
            raise ValueError("OpenAPI endpoint must not redirect")
        raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("OpenAPI export exceeds the 32 MiB capture limit")
        status = response.status
    json.loads(raw)  # Reject HTML/errors; retain the exact successful JSON bytes.
    provenance = json.loads(args.provenance.read_text())
    for key in ("source_revision", "server_version", "dependencies"):
        if not provenance.get(key):
            raise ValueError(f"Missing acquisition provenance: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "openapi.raw.json").write_bytes(raw)
    write_json(
        args.output_dir / "acquisition.json",
        {
            "schema": "protocol-http-acquisition/v1",
            "server": args.server,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "http_status": status,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "image": args.image,
            "container_id": container["Id"],
            "image_id": container["Image"],
            "entrypoint": container["Config"]["Entrypoint"],
            "command": container["Config"]["Cmd"],
            "ports": container["NetworkSettings"]["Ports"],
            "resources": {
                key: container["HostConfig"][key]
                for key in (
                    "NanoCpus",
                    "Memory",
                    "ShmSize",
                    "DeviceRequests",
                    "IpcMode",
                )
            },
            "provenance": provenance,
        },
    )
    return 0


def load_capture(directory: Path, server: str) -> tuple[dict, dict]:
    raw = (directory / "openapi.raw.json").read_bytes()
    metadata = json.loads((directory / "acquisition.json").read_text())
    if (
        metadata["schema"] != "protocol-http-acquisition/v1"
        or metadata["server"] != server
    ):
        raise ValueError(f"Wrong acquisition kind for {server}")
    if metadata["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"Raw {server} export changed after acquisition")
    return json.loads(raw), metadata
