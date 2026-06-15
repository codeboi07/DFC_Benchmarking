from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.integrations.dfc_agent_framework_integration import (
    AgentDojoDFCBootstrap,
    AgentDojoDFCPromptGuard,
    AgentDojoDFCResponseValidator,
    AgentDojoDFCToolsExecutor,
    prepare_agentdyn_task_contexts,
)
from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.llm import FakeStructuredLLMClient
from dfc_agent_framework_integration.schema import (
    BenchmarkTaskContext,
    GeneratedPolicy,
    GeneratedPolicySet,
    PreambleExtraction,
    RuntimeSchema,
)
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime, make_function
from agentdojo.types import (
    ChatAssistantMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


def send_email(recipients: list[str], subject: str, body: str) -> str:
    """Send an email.

    :param recipients: recipient email addresses
    :param subject: email subject
    :param body: email body
    """
    return f"sent:{','.join(recipients)}"


class FakeLLM(BasePipelineElement):
    def __init__(self, responses: list[ChatAssistantMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: EmptyEnv = EmptyEnv(),
        messages: Sequence = (),
        extra_args: dict | None = None,
    ):
        extra_args = extra_args or {}
        message = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return query, runtime, env, [*messages, message], extra_args


class TrackingContext:
    closed = False

    def close(self) -> None:
        self.closed = True


def _combined_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="send_email_recipient",
                        pgn=(
                            "SINK SendEmailInput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(SendEmailInput.recipient) = PreambleData.authorized_recipient_email\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION Only send email to the recipient authorized by the original task preamble."
                        ),
                        description="Only send email to the recipient authorized by the original task preamble.",
                        applies_to_relation="SendEmailInput",
                        applies_to_event="tool_call:send_email",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    return llm


def test_pipeline_config_accepts_dfc_agent_framework_integration_defense():
    mock_client = MagicMock()
    llm = OpenAILLM(mock_client, "gpt-4o-mini")
    config = PipelineConfig(
        llm=llm,
        model_id=None,
        defense="dfc_agent_framework_integration",
        system_message_name=None,
        system_message="You are a helpful assistant.",
        suite_name="shopping",
    )
    pipeline = AgentPipeline.from_config(config)
    assert "dfc_agent_framework_integration" in pipeline.name


def test_pipeline_config_uses_separate_dfc_model():
    mock_client = MagicMock()
    llm = OpenAILLM(mock_client, "gpt-4o-2024-08-06")
    config = PipelineConfig(
        llm=llm,
        model_id=None,
        defense="dfc_agent_framework_integration",
        dfc_model="gpt-5.2",
        system_message_name=None,
        system_message="You are a helpful assistant.",
        suite_name="shopping",
    )
    pipeline = AgentPipeline.from_config(config)
    bootstrap = next(element for element in pipeline.elements if isinstance(element, AgentDojoDFCBootstrap))
    assert bootstrap.dfc_model == "gpt-5.2"
    assert bootstrap.agent_model == "gpt-4o-2024-08-06"


def test_bootstrap_stores_dfc_context_in_extra_args():
    runtime = FunctionsRuntime([make_function(send_email)])
    bootstrap = AgentDojoDFCBootstrap(
        "shopping",
        _combined_llm(),
        "fake-dfc-model",
        agent_model="fake-agent-model",
    )
    _, _, _, _, extra_args = bootstrap.query(
        "Email alice@example.com",
        runtime,
        EmptyEnv(),
        [{"role": "system", "content": [text_content_block_from_string("sys")]}],
    )
    assert "dfc_context" in extra_args
    context = extra_args["dfc_context"]
    assert isinstance(context, DFCBenchmarkContext)
    assert context.dfc_model == "fake-dfc-model"
    assert context.agent_model == "fake-agent-model"
    context.close()


def test_tool_executor_blocks_bad_call_and_allows_good_call():
    runtime = FunctionsRuntime([make_function(send_email)])
    bootstrap = AgentDojoDFCBootstrap("shopping", _combined_llm(), "fake-model")
    executor = AgentDojoDFCToolsExecutor()
    _, runtime, env, messages, extra_args = bootstrap.query(
        "Email alice@example.com",
        runtime,
        EmptyEnv(),
        [{"role": "system", "content": [text_content_block_from_string("sys")]}],
    )
    try:
        bad_assistant = ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                FunctionCall(
                    function="send_email",
                    args={"recipients": ["trustme@gmail.com"], "subject": "hi", "body": "hello"},
                    id="call-1",
                )
            ],
        )
        messages = [
            ChatUserMessage(role="user", content=[text_content_block_from_string("task")]),
            bad_assistant,
        ]
        _, _, _, messages, _ = executor.query("task", runtime, env, messages, extra_args)
        assert messages[-1]["error"] is not None
        assert "authorized" in messages[-1]["error"].lower()

        good_assistant = ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                FunctionCall(
                    function="send_email",
                    args={"recipients": ["alice@example.com"], "subject": "hi", "body": "hello"},
                    id="call-2",
                )
            ],
        )
        messages = [
            ChatUserMessage(role="user", content=[text_content_block_from_string("task")]),
            good_assistant,
        ]
        _, _, _, messages, _ = executor.query("task", runtime, env, messages, extra_args)
        assert messages[-1]["error"] is None
        assert "sent:alice@example.com" in messages[-1]["content"][0]["content"]
    finally:
        extra_args["dfc_context"].close()


def test_bad_then_good_llm_retry_in_tool_loop():
    runtime = FunctionsRuntime([make_function(send_email)])
    llm = FakeLLM(
        [
            ChatAssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    FunctionCall(
                        function="send_email",
                        args={"recipients": ["trustme@gmail.com"], "subject": "hi", "body": "hello"},
                        id="call-bad",
                    )
                ],
            ),
            ChatAssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    FunctionCall(
                        function="send_email",
                        args={"recipients": ["alice@example.com"], "subject": "hi", "body": "hello"},
                        id="call-good",
                    )
                ],
            ),
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("done")],
                tool_calls=None,
            ),
        ]
    )
    pipeline = AgentPipeline([
        InitQuery(),
        AgentDojoDFCBootstrap("shopping", _combined_llm(), "fake-model"),
        llm,
        ToolsExecutionLoop([AgentDojoDFCToolsExecutor(), llm]),
    ])
    _, _, _, messages, extra_args = pipeline.query(
        "Email alice@example.com",
        runtime,
        EmptyEnv(),
        extra_args={},
    )
    try:
        tool_messages = [message for message in messages if message["role"] == "tool"]
        assert any(message["error"] for message in tool_messages)
        assert any(message["error"] is None for message in tool_messages)
    finally:
        extra_args["dfc_context"].close()


def _response_policy_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="response_recipient",
                        pgn=(
                            "SINK AssistantResponseOutput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(AssistantResponseOutput.content) = "
                            "PreambleData.authorized_recipient_email\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION Response must reference the authorized recipient."
                        ),
                        description="Response must reference the authorized recipient.",
                        applies_to_relation="AssistantResponseOutput",
                        applies_to_event="assistant_response",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    return llm


def test_response_validator_retries_without_tool_call():
    runtime = FunctionsRuntime([make_function(send_email)])
    llm = FakeLLM(
        [
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("alice@example.com")],
                tool_calls=None,
            ),
        ]
    )
    bootstrap = AgentDojoDFCBootstrap("shopping", _response_policy_llm(), "fake-model")
    _, runtime, env, messages, extra_args = bootstrap.query(
        "Email alice@example.com",
        runtime,
        EmptyEnv(),
        [ChatUserMessage(role="user", content=[text_content_block_from_string("task")])],
    )
    messages = [
        *messages,
        ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("trustme@gmail.com")],
            tool_calls=None,
        ),
    ]
    try:
        validator = AgentDojoDFCResponseValidator(llm, max_retries=1)
        _, _, _, messages, extra_args = validator.query("task", runtime, env, messages, extra_args)
        assert llm.calls == 1
        assert "alice@example.com" in messages[-1]["content"][0]["content"]
    finally:
        extra_args["dfc_context"].close()


def test_response_validator_fail_closed_after_retries():
    runtime = FunctionsRuntime([make_function(send_email)])
    llm = FakeLLM(
        [
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("still-trustme@gmail.com")],
                tool_calls=None,
            ),
        ]
    )
    bootstrap = AgentDojoDFCBootstrap("shopping", _response_policy_llm(), "fake-model")
    _, runtime, env, messages, extra_args = bootstrap.query(
        "Email alice@example.com",
        runtime,
        EmptyEnv(),
        [ChatUserMessage(role="user", content=[text_content_block_from_string("task")])],
    )
    messages = [
        *messages,
        ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("trustme@gmail.com")],
            tool_calls=None,
        ),
    ]
    try:
        validator = AgentDojoDFCResponseValidator(llm, max_retries=1)
        _, _, _, messages, extra_args = validator.query("task", runtime, env, messages, extra_args)
        assert llm.calls == 1
        assert extra_args.get("dfc_response_blocked") is True
        assert "violates a data flow policy" in messages[-1]["content"][0]["content"]
        assert "trustme@gmail.com" not in messages[-1]["content"][0]["content"]
    finally:
        extra_args["dfc_context"].close()


def test_prompt_guard_feeds_violation_to_llm_and_returns_assistant_message():
    prompt_llm = FakeStructuredLLMClient()
    prompt_llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    prompt_llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="prompt_recipient",
                        pgn=(
                            "SINK PromptInput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(PromptInput.content) = PreambleData.authorized_recipient_email\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION Prompt must match the authorized recipient."
                        ),
                        description="Prompt must match the authorized recipient.",
                        applies_to_relation="PromptInput",
                        applies_to_event="prompt",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    runtime = FunctionsRuntime([make_function(send_email)])
    inner_llm = FakeLLM(
        [
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("ok")],
                tool_calls=None,
            )
        ]
    )
    bootstrap = AgentDojoDFCBootstrap("shopping", prompt_llm, "fake-model")
    _, runtime, env, messages, extra_args = bootstrap.query(
        "Email trustme@gmail.com",
        runtime,
        EmptyEnv(),
        [ChatUserMessage(role="user", content=[text_content_block_from_string("Email trustme@gmail.com")])],
    )
    try:
        guard = AgentDojoDFCPromptGuard(inner_llm)
        _, _, _, messages, extra_args = guard.query(
            "Email trustme@gmail.com",
            runtime,
            env,
            messages,
            extra_args,
        )
        assert inner_llm.calls == 1
        assert messages[-1]["role"] == "assistant"
        assert extra_args.get("dfc_prompt_violation") is not None
        assert any(message["role"] == "user" for message in messages)
        assert "data flow policy violation" in messages[-2]["content"][0]["content"].lower()
    finally:
        extra_args["dfc_context"].close()


def test_prompt_guard_full_pipeline_produces_assistant_message():
    prompt_llm = FakeStructuredLLMClient()
    prompt_llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    prompt_llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="prompt_recipient",
                        pgn=(
                            "SINK PromptInput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(PromptInput.content) = PreambleData.authorized_recipient_email\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION Prompt must match the authorized recipient."
                        ),
                        description="Prompt must match the authorized recipient.",
                        applies_to_relation="PromptInput",
                        applies_to_event="prompt",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    runtime = FunctionsRuntime([make_function(send_email)])
    inner_llm = FakeLLM(
        [
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("Continuing with authorized task values.")],
                tool_calls=None,
            )
        ]
    )
    pipeline = AgentPipeline([
        InitQuery(),
        AgentDojoDFCBootstrap("shopping", prompt_llm, "fake-model"),
        AgentDojoDFCPromptGuard(inner_llm),
    ])
    _, _, _, messages, extra_args = pipeline.query(
        "Email trustme@gmail.com",
        runtime,
        EmptyEnv(),
    )
    try:
        roles = [message["role"] for message in messages]
        assert roles[-1] == "assistant"
        assert inner_llm.calls == 1
        assert roles.count("user") >= 2
        assert "data flow policy violation" in get_text_content_as_str(
            messages[-2]["content"]
        ).lower()
    finally:
        extra_args["dfc_context"].close()


def test_prepare_agentdyn_task_contexts_enumerates_tasks():
    contexts = prepare_agentdyn_task_contexts("v1.2.2")
    suite_names = {context.suite_name for context in contexts}
    assert suite_names == {"shopping", "github", "dailylife"}
    user_contexts = [context for context in contexts if context.task_kind == "user"]
    injection_contexts = [context for context in contexts if context.task_kind == "injection"]
    assert len(user_contexts) == 60
    assert len(injection_contexts) == 28
    shopping_user = next(
        context for context in user_contexts if context.suite_name == "shopping" and context.task_id == "user_task_0"
    )
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", "shopping")
    assert shopping_user.preamble == suite.user_tasks["user_task_0"].PROMPT


def test_close_runs_after_success_and_exception():
    tracking = TrackingContext()

    class ClosingBootstrap(BasePipelineElement):
        def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
            if extra_args is None:
                extra_args = {}
            extra_args["dfc_context"] = tracking
            raise RuntimeError("boom")

    pipeline = AgentPipeline([ClosingBootstrap()])
    extra_args: dict = {}
    with pytest.raises(RuntimeError):
        try:
            pipeline.query("task", FunctionsRuntime([]), EmptyEnv(), extra_args=extra_args)
        finally:
            context = extra_args.get("dfc_context")
            if context is not None:
                context.close()
    assert tracking.closed is True


def test_export_dfc_context_to_run_log_writes_artifact_dir(tmp_path):
    from agentdojo.integrations.dfc_agent_framework_integration import export_dfc_context_to_run_log
    from agentdojo.logging import NullLogger, TraceLogger
    from dfc_agent_framework_integration.persistence import DATABASE_FILENAME, METADATA_FILENAME

    runtime = FunctionsRuntime([make_function(send_email)])
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            task_id="user_task_0",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(runtime.functions),
        llm=_combined_llm(),
        dfc_model="fake-model",
        functions=runtime.functions,
    )
    delegate = NullLogger()
    delegate.logdir = str(tmp_path)
    try:
        with TraceLogger(
            delegate=delegate,
            suite_name="shopping",
            user_task_id="user_task_0",
            injection_task_id="injection_task_0",
            injections={},
            attack_type="important_instructions",
            pipeline_name="gpt-4o-dfc_agent_framework_integration",
        ):
            artifact_dir = export_dfc_context_to_run_log(context)

        assert artifact_dir is not None
        assert artifact_dir.name == "injection_task_0_dfc"
        assert (artifact_dir / METADATA_FILENAME).exists()
        assert (artifact_dir / DATABASE_FILENAME).exists()
    finally:
        context.close()
