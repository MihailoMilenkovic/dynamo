// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use utoipa::ToSchema;

#[test]
fn chat_only_fields_are_exported_without_advertising_them_for_completions() {
    let mut schemas = Vec::new();
    NvCreateChatCompletionRequest::schemas(&mut schemas);
    let chat_common = schemas
        .iter()
        .find(|(name, _)| name == "ChatCommonExt")
        .unwrap();
    let chat_schema = serde_json::to_value(&chat_common.1).unwrap();
    let properties = &chat_schema["allOf"][1]["properties"];
    for field in ["add_generation_prompt", "continue_final_message"] {
        assert_eq!(
            properties[field]["type"],
            serde_json::json!(["boolean", "null"])
        );
    }
    let shared = schemas
        .iter()
        .find(|(name, _)| name == "CommonExt")
        .unwrap();
    let shared = serde_json::to_value(&shared.1).unwrap();
    assert!(shared["properties"].get("add_generation_prompt").is_none());
    assert!(shared["properties"].get("continue_final_message").is_none());
}
