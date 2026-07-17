"""Minimal LiteLLM stand-in for an OpenAI client.

LiteLLM speaks the OpenAI chat-completions format in *and* out, so by exposing a `chat.completions.create`
method that forwards to `litellm.completion`, the existing `OpenAILLM` pipeline element can drive any
LiteLLM-routed model (e.g. a Bedrock Converse application-inference-profile such as Qwen3-235B) with no
changes to OpenAILLM itself. Per-model routing params (region, the profile ARN) are carried here.

If the env var `DFC_AGENT_IO_LOG` is set, every call's raw request + response (messages, tool calls,
usage, latency, errors) is appended there as JSONL — for post-mortem when a run misbehaves.
"""
from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import openai

_io_lock = threading.Lock()


def _serialize_response(resp) -> dict:
    try:
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return {
            "content": getattr(msg, "content", None),
            "tool_calls": [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (getattr(msg, "tool_calls", None) or [])
            ],
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (dict(usage) if usage else None),
        }
    except Exception as exc:  # never let logging break a run
        return {"serialize_error": repr(exc)}


def _log_io(path: str, request: dict, response, error, latency_s: float) -> None:
    rec = {
        "model": request.get("model"),
        "messages": request.get("messages"),
        "tools": [t.get("function", {}).get("name") for t in (request.get("tools") or [])],
        "response": _serialize_response(response) if response is not None else None,
        "error": error,
        "latency_s": round(latency_s, 2),
    }
    try:
        with _io_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


class _LiteLLMCompletions:
    def __init__(self, extra: dict) -> None:
        self._extra = extra  # routing params litellm needs, e.g. {"aws_region_name": "us-west-2"}

    def create(self, **kwargs):
        import litellm

        # OpenAILLM passes openai.NOT_GIVEN sentinels for omitted params — strip them, add the routing
        # params, and let litellm drop anything the target model doesn't support (e.g. reasoning_effort).
        clean = {k: v for k, v in kwargs.items() if v is not openai.NOT_GIVEN}
        clean.update(self._extra)
        clean["drop_params"] = True
        # Bound each call so a stalled Bedrock connection can't hang a worker forever (no timeout = the call
        # blocks with no error/retry/log). Successful agent calls are ~2s (p90 3s, slow tail ~36s), so 90s is
        # generous; on timeout litellm.completion raises and litellm.num_retries gets a fresh connection.
        clean.setdefault("timeout", float(os.getenv("DFC_QWEN_TIMEOUT", "90")))
        clean.setdefault("num_retries", int(os.getenv("DFC_QWEN_NUM_RETRIES", "6")))
        log_path = os.getenv("DFC_AGENT_IO_LOG")
        t0 = time.time()
        try:
            response = litellm.completion(**clean)
        except Exception as exc:
            if log_path:
                _log_io(log_path, clean, None, repr(exc), time.time() - t0)
            raise
        if log_path:
            _log_io(log_path, clean, response, None, time.time() - t0)
        return response


class LiteLLMClient:
    """Drop-in replacement for an OpenAI client object, backed by `litellm.completion`."""

    def __init__(self, **extra) -> None:
        self.chat = SimpleNamespace(completions=_LiteLLMCompletions(extra))
