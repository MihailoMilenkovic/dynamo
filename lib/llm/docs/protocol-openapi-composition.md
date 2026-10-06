<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B1 OpenAPI composition variant

Status: replacement workflow exercised against Docker-hosted Dynamo 1.6.0 and
vLLM 0.30.0. Tracking: DIS-3060. The tooling works with disclosed schema-fidelity
gaps; this is not a claim of complete compatibility coverage.

## Decision

Compare the request contracts exported by the two actual HTTP servers. Start
Dynamo and vLLM serve in pinned Docker images and retain each `/openapi.json`
response. Compose Dynamo's raw export with explicit imports from a pinned
OpenAI specification, then compare the resulting request contracts with vLLM's
independently acquired document. Chat completion is primary; completion is for
compatibility. Responses and behavioral conformance remain separate follow-ups.

The original B1 branch is a comparison baseline, not a production fallback.
This variant removes its superseded source parsers and reconstruction code.
Library-backed YAML loading, JSON Pointer, JSON Patch, schema validation, and
OpenAPI comparison replace custom parsing and comparison wherever practical.

## Dependency boundary

Keep Dynamo's runtime dependency versions: dynamo-protocols 6.1.0 and
async-openai 0.41.1. Adapt schema annotations from
[frontend-crates #338](https://github.com/ai-dynamo/frontend-crates/pull/338)
without its unrelated protocols 7 / async-openai 0.42 upgrade or dependency on
async-openai schema derives.

The annotated dependency exports owned schemas and explicit
`x-dynamo-schema-import` slots for unannotated async-openai types. A slot is not
a complete contract, even though the raw document remains syntactically usable.
The composition stage must resolve every in-scope slot or fail visibly.

Use the OpenAI YAML from async-openai's published 0.41.1 source revision
`884aff958c0461cce41e2d9e9b2fe4f29e76b740`:

- File: `openapi.yaml`.
- SHA-256: `74cbcf73838f4cd7e209b2d3f2e9ddc9fa155f21a44360b6fac7646a6d4f5f8b`.
- This pin establishes provenance, not automatic equivalence with Rust types.

## Composition invariants

1. Preserve raw inputs; produce a separate composed Dynamo document.
2. Import only explicitly selected definitions and their reference closure.
3. Preserve Dynamo endpoints and owned fields. Never fill gaps from vLLM.
4. Namespace imported components; reject collisions and unmapped slots.
5. Express corrections with standard JSON Patch. Guard each change with a
   preceding `test` of the affected value (or parent for additions), and record
   the Rust-code rationale. A failed guard requires review, not silent repair.
6. Redirect references to corrected definitions, so nested uses cannot bypass
   their corrections. Preserve recursion as references, not infinite expansion.
7. Do not fetch external references. Reject unsupported resolution constructs.
8. Walk schema positions only; example/default payloads may contain literal
   `$ref` keys and must remain data.

An explicit `serde_json::Value` field is intentionally untyped, not necessarily
an extraction defect. In contrast, an unresolved dependency marker is a gap.
The contents of `nvext` may be excluded because they are Dynamo-specific;
record that exclusion, never label it compatible, and do not generalize it to
other fields without evidence.

## HTTP acquisition

The existing `generate-frontend-openapi` binary has a `--serve
IP:PORT` mode. It runs Dynamo's actual `HttpService` router with its normal
OpenAPI handler, in-memory discovery, enabled endpoint families and no workers.
It is not a file server serving a saved schema. No model or GPU is needed for
this Dynamo-side acquisition; inference and readiness are not tested by it.

For vLLM, use its real serve command and query its HTTP OpenAPI endpoint. Record
its actual startup requirements and configuration; do not substitute standalone
Pydantic export without declaring a scope change.

The Docker packaging recipe is
`scripts/protocol_compatibility/acquisition/Dockerfile.dynamo`. Build the helper
on Linux from the exact checkout with the pinned Cargo lock, then package a
stripped copy of the binary. Preserve the compiler version, command, source
revision, any working-tree patch, lockfile and both binary hashes. This recipe
packages a previously built binary; it does not attest a reproducible source build.

```sh
cargo build --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
mkdir image-context
cp target/debug/generate-frontend-openapi image-context/generate-frontend-openapi
strip image-context/generate-frontend-openapi
cp scripts/protocol_compatibility/acquisition/Dockerfile.dynamo image-context/Dockerfile
docker build --platform linux/amd64 --provenance=false -t "$DYNAMO_IMAGE_TAG" image-context
```

Retain the built image in the approved registry, verify its remote manifest
digest, and set `DYNAMO_IMAGE` to that repository@sha256 reference. Do not confuse
an image's configuration digest with its pullable manifest digest.

```sh
docker run -d --name "$DYNAMO_CONTAINER" -p 127.0.0.1::8000 "$DYNAMO_IMAGE"
docker port "$DYNAMO_CONTAINER" 8000/tcp
```

The verified vLLM startup used v0.30.0, a GPU, cached Qwen/Qwen3-0.6B weights,
snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, and eager execution. Set
`MODEL_CACHE` to the model-specific Hugging Face cache directory containing both
`snapshots/` and `blobs/`; mount it read-only. Do not mount credentials or an entire
home directory. This is schema acquisition, not a benchmark.

```sh
VLLM_IMAGE=vllm/vllm-openai@sha256:439c19d48db36401abc914b9842d060fe610a54bc3ac29bb02505f0b69af1baf
docker run -d --name "$VLLM_CONTAINER" --gpus device=0 \
  --cpus 6 --memory 16g --shm-size 2g -p 127.0.0.1::8000 \
  -v "$MODEL_CACHE:/model-cache:ro" -e HF_HUB_OFFLINE=1 "$VLLM_IMAGE" \
  /model-cache/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --served-model-name protocol-schema-probe --max-model-len 512 \
  --gpu-memory-utilization 0.15 --enforce-eager --host 0.0.0.0 --port 8000
docker port "$VLLM_CONTAINER" 8000/tcp
```

Wait for each server's actual `/openapi.json` to return HTTP 200, then use the
[capture commands](../../../scripts/protocol_compatibility/README.md). Record
the dynamic ports instead of assuming a host port. vLLM's `/version` provides an
independent version check. Stop/remove only these named task containers afterward;
verify that GPU memory is released. Keep the captured specs and retained image.

### Acquisition provenance

Supply a JSON provenance file per server. Required keys are `source_revision`,
`server_version`, and `dependencies`. For Dynamo, the dependency object must
exactly match the reviewed composition manifest:

```json
{
  "source_revision": "<full Dynamo commit SHA>",
  "server_version": "1.6.0",
  "dependencies": {
    "async-openai": "0.41.1",
    "dynamo-protocols": "6.1.0",
    "dynamo-protocols-revision": "c5e29b8db2f175ac7709f260d58a2fb9404f98bd"
  },
  "build": {
    "source_state_sha256": "<exact working-state digest>",
    "binary_sha256": "<packaged binary digest>",
    "command": "cargo build --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi"
  }
}
```

For vLLM, record its actual version/source revision if available, the image digest
and version evidence, and dependency/model/configuration information used at
startup. A version tag is not a source commit: label it accurately if the image
does not expose a commit. The image digest still pins the executed artifact.
The capture command verifies the running container's requested image and published
port; it cannot independently prove user-supplied source/build provenance.

The workflow copies raw captures, composition YAML, pinned OpenAI YAML, scoped
request documents, the complete library diff and Markdown/JSON reports into a
fresh assessment directory. It never writes reports into this repository.

## Comparison boundaries and known gaps

Only the two POST JSON request bodies and their schema reference closure reach
the comparator. The projection retains constraints and local references, removes
unrelated endpoints/response types, and validates the scoped OpenAPI document.
Raw and composed full documents remain separate retained inputs.

The current full raw Dynamo document has an unrelated batch-route documentation
defect (a missing `batch_id` path parameter). Scoped validation does not claim to
validate every endpoint in that full document.

The comparator is pinned to oasdiff 1.33.0, with external reference fetching
disabled. It performs dialect normalization and allOf flattening only when the
coverage scan finds no unsupported constructs. Unsupported constructs stay in
the inputs and are reported as gaps; they are not silently removed. The library's
[allOf limitations](https://github.com/oasdiff/oasdiff/blob/v1.33.0/docs/ALLOF.md)
motivate these guards. The tool is not a general JSON Schema equivalence prover.
Equivalent union/nullability representations can still produce structural diff
candidates. Review the complete request-body delta before calling one a wire
incompatibility. Added/deleted refer to vLLM → Dynamo, not an upstream upgrade.

`assessment/coverage.yaml` records known native-export limitations, including
Serde aliases, object-valued function arguments, conditional system-message
content, role restrictions, empty-media normalization, and incompletely declared
numeric/custom validation. These are **coverage gaps**, not intentional exclusions
or approved compatibility. They currently prevent a complete-coverage claim.

The explicit `nvext` exclusion concerns its internal Dynamo-only contents. Its
top-level presence may still appear as a Dynamo-only declaration. Other extension
fields are not excluded simply because their schemas are difficult to resolve.

## Maintenance and acceptance

On dependency updates, review the import mappings and corrections against the
new Rust types, pin the corresponding YAML revision, rerun schema/Serde fidelity
cases, and reacquire both HTTP exports. Framework-version changes use the same
direct Dynamo-versus-vLLM workflow; comparing old/new vLLM alone is not the goal.

The version-bump procedure is:

1. Record the new Dynamo commit/lock and target framework image digest/version.
2. For an async-openai or protocols change, review every import mapping and native
   schema override against the new Rust types. Update pins and guarded patches;
   never just refresh a checksum to bypass a failing guard.
3. Review the coverage catalog. Remove a gap only with a schema fix and a fidelity
   case proving the newly represented input shape; do not remove it to green a gate.
4. Build the exact Dynamo image and reacquire both HTTP specs under recorded
   configuration. Model/task/flags can affect available endpoints or schemas.
5. Run the Rust cases, validate the composed schema against the same cases, and
   produce the direct assessment report. Attach the revision-pinned report to
   the version-bump review and track newly discovered differences explicitly.
6. Compare retained reports for review context, without replacing the direct
   Dynamo-versus-target assessment with an upstream-only diff.

The CI workflow checks the Python pipeline and real pinned comparator on relevant
PRs, including framework/dependency pin changes. It does **not** run GPU server
acquisition or approve protocol parity. The old source-based weekly watcher and
candidate/source-pin commands are retired in this variant. Periodic runtime
capture scheduling requires an appropriately provisioned runner; no such hosted
GPU automation is claimed here. The explicit version-bump procedure remains
mandatory regardless of a green tooling check.

For periodic maintenance, the frontend protocol owner should review upstream
releases weekly, select an immutable candidate image when the target changes,
and run this same direct capture/assessment procedure. Record a no-change review
or attach the candidate report and create owned follow-up issues for new gaps or
differences. Reviewing a candidate must not automatically adopt its image pin.
This is a documented human-run procedure, not an installed background monitor.

## Replacement accounting

The original B1 branch at `e561f20a2c424d4683a2e4d8bef43184874fe6b1` remains the
comparison baseline. This variant removes its Rust/Python source interpreters,
dependency scanner, reconstructed contracts, historical adapters, unused generated
vocabulary, source-pin catalogs and source-specific triage/history implementation.
It retains no fallback source parser. The historical N-2 fixture generator and
its unused base-branch fixtures are removed; this is not a change to a runtime
mixed-version compatibility promise or proof of mixed-version behavior.

Old tests of parser internals, historical extraction, inventory freshness and
retired command/triage interfaces are removed with those features. Replacement
tests cover reference closure, nested field/type/constraint changes, requiredness,
nullability, malformed inputs, explicit gaps, provenance checks, guarded imports,
request-only scope and shared Rust/schema fidelity. This is coverage of the new
mechanism, not a claim that test counts must match. Final LoC accounting must
include the dependency annotation changes and manifests as well as Python/Rust
implementation and tests; moving code out of the tree is not a reduction by itself.

Report declared differences, coverage gaps and intentional exclusions separately.
Schema agreement does not establish forwarding, generation, streaming or other
behavioral conformance. Unsupported comparison constructs must remain visible.

Before calling the variant complete, demonstrate the real assessment, remove
superseded B1 machinery and stale workflows/docs, account for implementation and
test LoC across both repositories, and retain revision-pinned validation evidence.

### Measured maintenance footprint

The following counts use physical source lines, including comments and blanks,
not semantic statements or Git diff additions. Python counts include package
initializers and test helpers. Compare B1 `e561f20a2c42` against this variant;
dependency additions compare protocols 6.1.0 `c788fb2ecb52` against the pinned
schema backport `c5e29b8db2f1`.

| Source category | B1 | Variant or additional cost | Net change |
|---|---:|---:|---:|
| Python tooling implementation | 4,066 | 875 | -3,191 |
| Python tests and fidelity helper | 2,339 | 832 | -1,507 |
| Dynamo Rust implementation | Existing source | 52 net new lines | +52 |
| Dependency Rust schema implementation | No schema support | 225 new lines | +225 |
| Dynamo Rust tests | No dedicated schema tests | 80 new lines | +80 |
| Dependency Rust schema tests | No schema tests | 171 new lines | +171 |

Across these implementation categories the net reduction is **2,914 lines**;
across test code it is **1,256 lines**. Separately, the variant removes 79 lines
of unused generated Rust vocabulary. These are not counted as handwritten
implementation savings.

Maintenance also includes 208 lines of import mappings/corrections, 44 lines of
known-gap catalog, 61 lines of shared test-case data, the 25-line Docker recipe,
dependency pins/lockfiles, CI configuration and documentation. The dependency
backport adds 180 lockfile lines, six Cargo manifest lines and 39 README lines;
those are not hidden inside the source-code totals. No upstream schema support
is added to async-openai itself.

Remaining custom implementation selects the two request contracts, checks
coverage/provenance, maps explicit imports, orchestrates library calls and
renders the report. It does not parse Rust/Python sources or implement a schema
equivalence engine. The YAML corrections remain a reviewed maintenance cost.

### Concrete assessment and validation boundary

The first final-image HTTP assessment used Dynamo Rust-support revision
`8a5e8f46740d074b1fa9cb04bed600c95cbba599` plus the recorded tooling patch, and
vLLM revision `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (0.30.0).
The raw Dynamo document has SHA-256
`b30d8551b38cfb165fc4a07badff98a2230fc7f506e24996e5290fe696016d10`;
the raw vLLM document has SHA-256
`b459a7137b8e9e54326935977ceff229b06502681dda07fa3593b6d76945e204`.
The archived acquisition records contain the image digests and exact commands.

The report contains two endpoint-level request deltas and seven known coverage
gaps, so assessment exits 1 (`incomplete_coverage`). These two deltas are groups,
not two individual incompatibilities. Chat has 31 vLLM-only declarations, 19
Dynamo-only declarations and 36 changed top-level fields; completion has 26,
nine and 27 respectively. Nested deltas remain in the complete JSON report.
For example, `chat_template_kwargs` appears vLLM-only in the schema even though
Dynamo accepts it as an alias: the report explicitly flags that export gap.

All 19 external OpenAI import slots resolve. Shared fidelity cases establish 48
matching schema/Serde outcomes and reproduce six intentional test witnesses of
known export gaps. They do not establish that every accepted request matches
the schema. The raw full API also retains the unrelated batch-route defect
described above; only the scoped request projection is validated.

This evidence covers the recorded text-generation configuration, not a matrix
of every model/task/flag combination. Reacquire under a different configuration
before extending the claim to it. No response or behavioral parity is inferred.
