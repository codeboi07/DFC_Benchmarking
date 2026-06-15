"""LLM pipeline element for non-Anthropic AWS Bedrock models via the Bedrock Converse API.

The repo's `bedrock` provider only drives Claude (through the Anthropic SDK). This element uses
boto3 `bedrock-runtime.converse()`, whose tool-use format is uniform across Bedrock models
(Llama 3.1+, GPT-OSS, etc.), so AgentDojo's tool-calling loop works with them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

import boto3
import jsonref

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    MessageContentBlock,
    TextContentBlock,
    get_text_content_as_str,
)

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _as_obj(value: Any) -> dict:
    """Coerce a tool-call input into a plain JSON object (dict) for the Converse API.

    GPT-OSS (a reasoning model) occasionally returns toolUse.input as a non-object (string, etc.);
    Converse rejects that on replay. Always hand back a plain dict so replay never fails on format.
    """
    if isinstance(value, dict):
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return {str(k): v for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _tool_config(functions: dict[str, Function]) -> dict | None:
    tools = []
    for fn in functions.values():
        schema = json.loads(json.dumps(jsonref.replace_refs(fn.parameters.model_json_schema(), proxies=False)))
        schema.pop("$defs", None)
        tools.append({"toolSpec": {"name": fn.name, "description": fn.description or fn.name, "inputSchema": {"json": schema}}})
    return {"tools": tools} if tools else None


def _to_converse_messages(messages: Sequence[ChatMessage]) -> tuple[list[dict] | None, list[dict]]:
    system_prompt: list[dict] | None = None
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_prompt = [{"text": get_text_content_as_str(m["content"])}]
        elif role == "user":
            out.append({"role": "user", "content": [{"text": get_text_content_as_str(m["content"])}]})
        elif role == "assistant":
            content: list[dict] = []
            text = get_text_content_as_str(m["content"]) if m["content"] else ""
            if text:
                content.append({"text": text})
            for tc in m["tool_calls"] or []:
                content.append({"toolUse": {"toolUseId": tc.id, "name": tc.function, "input": _as_obj(tc.args)}})
            if not content:
                content = [{"text": ""}]
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            result_text = m["error"] if m["error"] is not None else get_text_content_as_str(m["content"])
            out.append({"role": "user", "content": [{
                "toolResult": {
                    "toolUseId": m["tool_call_id"],
                    "content": [{"text": result_text or ""}],
                    "status": "error" if m["error"] is not None else "success",
                }
            }]})
    # Converse requires alternating roles; merge consecutive same-role (user) messages into one.
    merged: list[dict] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})
    return system_prompt, merged


def _parse_converse_response(message: dict) -> ChatAssistantMessage:
    text_blocks: list[MessageContentBlock] = []
    tool_calls: list[FunctionCall] = []
    for block in message.get("content", []):
        if "text" in block:
            text_blocks.append(TextContentBlock(type="text", content=block["text"]))
        elif "toolUse" in block:
            tu = block["toolUse"]
            if _TOOL_NAME_RE.match(tu["name"]):
                tool_calls.append(FunctionCall(id=tu["toolUseId"], function=tu["name"], args=_as_obj(tu.get("input"))))
    return ChatAssistantMessage(
        role="assistant",
        content=text_blocks or [TextContentBlock(type="text", content="")],
        tool_calls=tool_calls or None,
    )


class BedrockConverseLLM(BasePipelineElement):
    """Drives a Bedrock model (by model id or inference-profile ARN) via the Converse API."""

    def __init__(self, model: str, region: str = "us-east-1", temperature: float | None = 0.0,
                 max_tokens: int = 4096, max_retries: int = 5) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _converse(self, system_prompt, messages, tool_config):
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature or 0.0},
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tool_config:
            kwargs["toolConfig"] = tool_config
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.converse(**kwargs)
            except Exception as exc:  # noqa: BLE001 - retry throttling / transient Bedrock errors
                last_exc = exc
                name = type(exc).__name__
                if "Throttling" in name or "TooManyRequests" in name or "ServiceUnavailable" in name:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        system_prompt, converse_messages = _to_converse_messages(messages)
        tool_config = _tool_config(runtime.functions)
        try:
            response = self._converse(system_prompt, converse_messages, tool_config)
            output = _parse_converse_response(response["output"]["message"])
        except Exception as exc:  # noqa: BLE001
            # A non-throttle Bedrock/Converse error (e.g. a malformed toolUse on replay) must not crash
            # the whole suite. End this rollout with an empty assistant turn so the task is scored
            # (as a failure) and the benchmark moves on. Logged so affected tasks are identifiable.
            logging.warning("BedrockConverse query failed; ending rollout gracefully: %s", str(exc)[:300])
            output = ChatAssistantMessage(
                role="assistant", content=[TextContentBlock(type="text", content="")], tool_calls=None
            )
        return query, runtime, env, [*messages, output], extra_args
