# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.protocol_compatibility.acquisition.http import (
    acquire,
    load_capture,
    write_json,
)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        provenance = root / "provenance.json"
        write_json(
            provenance,
            {
                "source_revision": "a" * 40,
                "server_version": "fixture",
                "dependencies": {"fixture": "1"},
            },
        )
        self.args = SimpleNamespace(
            server="dynamo",
            container="fixture-container",
            image="fixture@sha256:" + "0" * 64,
            url="http://127.0.0.1:12345/openapi.json",
            provenance=provenance,
            output_dir=root / "capture",
        )
        self.container = {
            "HostConfig": {
                "NanoCpus": 0,
                "Memory": 0,
                "ShmSize": 0,
                "DeviceRequests": None,
                "IpcMode": "private",
            },
            "Id": "container-id",
            "Image": "image-id",
            "State": {"Running": True},
            "Config": {
                "Image": self.args.image,
                "Entrypoint": ["server"],
                "Cmd": [],
                "Env": ["SECRET=must-not-be-copied"],
            },
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "12345", "HostIp": "127.0.0.1"}]}
            },
        }

    @patch("scripts.protocol_compatibility.acquisition.http.build_opener")
    @patch("scripts.protocol_compatibility.acquisition.http.subprocess.run")
    def test_capture_preserves_raw_bytes_and_omits_environment(self, run, opener):
        run.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps([self.container])
        )
        response = MagicMock()
        response.url, response.status = self.args.url, 200
        response.read.return_value = b'{"openapi": "3.1.0"}\n'
        opener.return_value.open.return_value.__enter__.return_value = response
        self.assertEqual(acquire(self.args), 0)
        raw = (self.args.output_dir / "openapi.raw.json").read_bytes()
        self.assertEqual(raw, response.read.return_value)
        metadata = (self.args.output_dir / "acquisition.json").read_text()
        self.assertNotIn("SECRET", metadata)
        self.assertNotIn("Env", metadata)
        self.assertEqual(
            load_capture(self.args.output_dir, "dynamo")[0]["openapi"], "3.1.0"
        )

    @patch("scripts.protocol_compatibility.acquisition.http.subprocess.run")
    def test_wrong_container_or_port_fails_before_http(self, run):
        for changed in ("image", "port", "running"):
            with self.subTest(changed=changed):
                container = json.loads(json.dumps(self.container))
                if changed == "image":
                    container["Config"]["Image"] = "mutable:latest"
                elif changed == "port":
                    container["NetworkSettings"]["Ports"] = {}
                else:
                    container["State"]["Running"] = False
                run.return_value = subprocess.CompletedProcess(
                    [], 0, json.dumps([container])
                )
                with self.assertRaises(ValueError):
                    acquire(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_mutable_image_remote_url_and_credentials_rejected(self):
        self.args.image = "fixture:latest"
        with self.assertRaisesRegex(ValueError, "immutable"):
            acquire(self.args)
        self.args.image = "fixture@sha256:" + "0" * 64
        for url in (
            "https://example.com/openapi.json",
            "http://user:secret@127.0.0.1:12345/openapi.json",
            "http://127.0.0.1:12345/openapi.json?token=secret",
        ):
            self.args.url = url
            with self.assertRaises(ValueError):
                acquire(self.args)


if __name__ == "__main__":
    unittest.main()
