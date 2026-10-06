# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Python-backend embedder: returns the first 8 FP32 values of
sha256(input text), so tests can regenerate the expected vector in-process."""

import hashlib

import numpy as np
import triton_python_backend_utils as pb_utils

_EMBEDDING_DIM = 8


def _embed(raw: object) -> np.ndarray:
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = np.frombuffer(digest, dtype="<f4", count=_EMBEDDING_DIM).copy()
    # Some sha256 digests land on NaN/Inf FP32 bit patterns, which the
    # OpenAI float path serializes as JSON null; clamp to finite so
    # arbitrary input text produces a valid embedding wire value.
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            # Batched shape is [batch, 1]; the trailing 1 is the per-item dim.
            batch_size = texts.shape[0]
            vectors = np.empty((batch_size, _EMBEDDING_DIM), dtype=np.float32)
            for i in range(batch_size):
                vectors[i] = _embed(texts[i, 0])
            responses.append(
                pb_utils.InferenceResponse([pb_utils.Tensor("EMBEDDING", vectors)])
            )
        return responses
