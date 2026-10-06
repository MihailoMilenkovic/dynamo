# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: dynamo.triton serves /v1/embeddings through the
HTTP frontend using the bundled Python-backend `fake_embedder` fixture."""

import base64
import hashlib
import json
import logging
import os
import shutil

import numpy as np
import pytest
import requests

pytest.importorskip("tritonserver")

from dynamo.triton.util import endpoint_slug  # noqa: E402
from tests.utils.managed_process import ManagedProcess  # noqa: E402

logger = logging.getLogger(__name__)

# The worker is pointed at a per-test copy of this fixture alone;
# ``--task=embed`` would reject the shared ``identity`` fixture next to it.
_FIXTURE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "components",
        "src",
        "dynamo",
        "triton",
        "models",
        "fake_embedder",
    )
)
_MODEL_NAME = "fake_embedder"
_EMBEDDING_DIM = 8
_WORKER_TIMEOUT_SECS = 300


def _expected_vector(text: str) -> np.ndarray:
    """Mirror of the fixture's embedding function so assertions stay bit-exact."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = np.frombuffer(digest, dtype="<f4", count=_EMBEDDING_DIM).copy()
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _decode_base64(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded), dtype="<f4")


class EmbedWorkerProcess(ManagedProcess):
    """Embed-task Triton worker bound to a single-model repository."""

    def __init__(
        self, request, system_port: int, log_dir: str, model_repo: str
    ) -> None:
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.triton",
            "--model-repository",
            model_repo,
            "--task",
            "embed",
            "--allow-metrics",
            "false",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = json.dumps(
            [endpoint_slug(_MODEL_NAME)]
        )
        env["DYN_SYSTEM_PORT"] = str(system_port)

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.json().get("status") == "ready",
                )
            ],
            timeout=_WORKER_TIMEOUT_SECS,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


@pytest.fixture(scope="function")
def start_services_with_embed_worker(request, tmp_path, start_services_with_http):
    """HTTP frontend + embed worker on a per-test single-model copy of the fixture."""
    frontend_port, system_port = start_services_with_http
    model_repo = tmp_path / "model_repo"
    shutil.copytree(_FIXTURE_DIR, model_repo / _MODEL_NAME)
    log_dir = str(tmp_path / f"{request.node.name}_worker")
    with EmbedWorkerProcess(request, system_port, log_dir, str(model_repo)):
        yield frontend_port


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_embeddings_float_round_trip(
    file_storage_backend, start_services_with_embed_worker
):
    """Two-item batch returns bit-exact per-item vectors in request order."""
    frontend_port = start_services_with_embed_worker

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/embeddings",
        json={"model": _MODEL_NAME, "input": ["hello", "world"]},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == _MODEL_NAME
    assert [item["index"] for item in body["data"]] == [0, 1]

    hello = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
    world = np.asarray(body["data"][1]["embedding"], dtype=np.float32)
    assert hello.shape == (_EMBEDDING_DIM,)
    assert world.shape == (_EMBEDDING_DIM,)
    np.testing.assert_array_equal(hello, _expected_vector("hello"))
    np.testing.assert_array_equal(world, _expected_vector("world"))


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_embeddings_base64_round_trip(
    file_storage_backend, start_services_with_embed_worker
):
    """``encoding_format=base64`` returns a lossless little-endian FP32 string."""
    frontend_port = start_services_with_embed_worker

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/embeddings",
        json={
            "model": _MODEL_NAME,
            "input": "hello",
            "encoding_format": "base64",
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1
    encoded = data[0]["embedding"]
    assert isinstance(encoded, str)
    np.testing.assert_array_equal(_decode_base64(encoded), _expected_vector("hello"))


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_embeddings_rejects_every_unsupported_control(
    file_storage_backend, start_services_with_embed_worker
):
    """Each unsupported control returns a structured 400 individually, so
    dropping any one from the reject-list or replacing the structured error
    with an HTML page fails this test."""
    frontend_port = start_services_with_embed_worker

    for field, value in (
        ("dimensions", 4),
        ("add_special_tokens", True),
        ("truncate_prompt_tokens", 128),
    ):
        response = requests.post(
            f"http://localhost:{frontend_port}/v1/embeddings",
            json={"model": _MODEL_NAME, "input": "hi", field: value},
            timeout=30,
        )
        assert response.status_code == 400, (
            f"expected 400 for {field}={value!r}, got {response.status_code}: "
            f"{response.text}"
        )
        body = response.json()
        assert (
            body.get("code") == 400
        ), f"expected structured 400 body for {field}={value!r}, got {body!r}"
