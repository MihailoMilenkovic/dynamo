<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo versus vLLM protocol assessment

This B1 variant replaces source-based extraction with HTTP OpenAPI acquisition,
explicit Dynamo/OpenAI schema composition, and library-backed request comparison.

See the [current design and reproduction guide](protocol-openapi-composition.md)
and the [tooling commands](../../../scripts/protocol_compatibility/README.md).

Declared request contracts and behavioral conformance are separate acceptance
layers. Optional source investigation supports diagnosis; it is not another
sequential acceptance gate. This variant implements the request-contract layer.

The original B1 source workflow and its evidence remain on
`codex/vllm-protocol-tooling` at the comparison baseline
`e561f20a2c424d4683a2e4d8bef43184874fe6b1`. Its source inventories, historical
adapters and triage commands are not retained as a fallback in this variant.
