<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protocol compatibility tooling

Compare **Dynamo versus vLLM serve**, using independently acquired HTTP OpenAPI
documents. Chat completion is primary; completion is included for compatibility.
Only declared JSON request contracts are assessed. Responses and behavioral
conformance are separate work.

Read the [composition and acquisition guide](../../lib/llm/docs/protocol-openapi-composition.md)
for Docker startup, provenance, schema corrections, limitations and maintenance.
Start with the generated `report.md`; retain `report.json` and input artifacts.

## Workflow

1. Start each server from its pinned Docker image and wait for HTTP readiness.
2. Use `acquire` to capture its real `/openapi.json` and container/source metadata.
3. Use `assess` to compose Dynamo with the pinned OpenAI YAML and compare requests.
4. Review differences, coverage gaps and intentional exclusions separately.

On a Linux validation host, install the pinned Python requirements and
[oasdiff 1.33.0](https://github.com/oasdiff/oasdiff/releases/tag/v1.33.0).
CI records the Linux amd64 release checksum; verify the appropriate release
checksum when using another platform.

```sh
python -m pip install -r scripts/protocol_compatibility/requirements.txt
python -m scripts.protocol_compatibility acquire --help
python -m scripts.protocol_compatibility assess --help

python -m scripts.protocol_compatibility acquire \
  --server dynamo --container "$DYNAMO_CONTAINER" --image "$DYNAMO_IMAGE" \
  --url "http://127.0.0.1:$DYNAMO_PORT/openapi.json" \
  --provenance dynamo-provenance.json --output-dir captures/dynamo

python -m scripts.protocol_compatibility acquire \
  --server vllm --container "$VLLM_CONTAINER" --image "$VLLM_IMAGE" \
  --url "http://127.0.0.1:$VLLM_PORT/openapi.json" \
  --provenance vllm-provenance.json --output-dir captures/vllm

python -m scripts.protocol_compatibility assess \
  --dynamo captures/dynamo --vllm captures/vllm \
  --openai openai-884aff95.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment
```

Output directories must not already exist. Input bytes and metadata are retained;
changed checksums, dependency-pin mismatches and unresolved imports fail visibly.
`acquire` does not start or stop containers; the caller owns their lifecycle.
It never records the container's environment variables. Review command arguments
and provenance files for secrets before publication.

Exit codes: `0` = no reported declared differences or known gaps; `1` = differences
or coverage gaps require review; `2` = invalid inputs or tooling failure. None
means runtime parity. The current coverage catalog intentionally prevents a
complete-coverage claim until the documented export limitations are resolved.

## Organization

| Directory | Responsibility |
| --- | --- |
| `acquisition/` | HTTP capture and Docker identity/provenance; Dynamo image recipe |
| `composition/` | Pinned OpenAI imports and checked spec-to-Rust corrections |
| `assessment/` | Request projection, coverage guards and oasdiff orchestration |
| `reporting/` | Developer-readable report |
| `tests/` | Unit/integration checks and shared Rust/schema fidelity cases |

No source parser or historical-commit fallback remains. The original B1 branch
is the historical baseline, not an alternative execution mode.

## Validation

```sh
OASDIFF_BIN=/absolute/path/to/oasdiff \
  python -m unittest discover -s scripts/protocol_compatibility/tests -t . -v
cargo test --locked -p dynamo-llm --no-default-features \
  --test openapi_request_schema --test openapi_request_fidelity
python -m scripts.protocol_compatibility.tests.composition.fidelity \
  --spec assessment/dynamo.composed.json
```

The real-comparator tests explicitly skip without `OASDIFF_BIN`; that skip is not
validation evidence. CI supplies the verified binary. Docker acquisition and
Rust/schema fidelity are additional checks, not implied by the Python unit suite.
