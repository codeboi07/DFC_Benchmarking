from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime, make_function
from agentdojo.integrations.dfc_agent_framework_integration import (
    AgentDojoDFCBootstrap,
    AgentDojoDFCToolsExecutor,
)
from agentdojo.types import (
    ChatAssistantMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from dfc_agent_framework_integration.llm import FakeStructuredLLMClient
from dfc_agent_framework_integration.schema import (
    GeneratedPolicy,
    GeneratedPolicySet,
    PreambleExtraction,
)
from dfc_agent_framework_integration.sql_tool import SQL_TOOL_NAME


class Contact(BaseModel):
    account: str
    name: str


def get_contacts() -> Contact:
    """Get the user's saved payee contact."""
    return Contact(account="DE123", name="Alice")


def send_money(recipient: str, amount: str) -> str:
    """Send money to a recipient.

    :param recipient: the payee account number
    :param amount: the amount to send
    """
    return f"sent {amount} to {recipient}"


def _runtime() -> FunctionsRuntime:
    return FunctionsRuntime([make_function(send_money), make_function(get_contacts)])


def _sr_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(PreambleExtraction, [PreambleExtraction.from_dict({"payee": "Alice"})])
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="send_money_recipient",
                        pgn=(
                            "SOURCE REQUIRED GetContactsOutput AS T SINK SendMoneyInput AS S\n"
                            "CONSTRAINT S.recipient = T.account\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION The recipient must be read from your saved contacts."
                        ),
                        description="The recipient must be read from your saved contacts.",
                        applies_to_relation="SendMoneyInput",
                        applies_to_event="tool_call:send_money",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    return llm


def _dimension_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(PreambleExtraction, [PreambleExtraction.from_dict({"payee": "Alice"})])
    llm.register(GeneratedPolicySet, [GeneratedPolicySet(policies=[])])
    return llm


def _bootstrap(llm):
    runtime = _runtime()
    bootstrap = AgentDojoDFCBootstrap("shopping", llm, "fake-model")
    _, runtime, env, messages, extra_args = bootstrap.query(
        "Send money to Alice",
        runtime,
        EmptyEnv(),
        [{"role": "system", "content": [text_content_block_from_string("sys")]}],
    )
    return runtime, env, messages, extra_args


def _assistant(function: str, args: dict, call_id: str) -> ChatAssistantMessage:
    return ChatAssistantMessage(
        role="assistant", content=None,
        tool_calls=[FunctionCall(function=function, args=args, id=call_id)],
    )


class FakeLLM(BasePipelineElement):
    def __init__(self, responses: list[ChatAssistantMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def query(self, query, runtime, env=EmptyEnv(), messages: Sequence = (), extra_args: dict | None = None):
        extra_args = extra_args or {}
        message = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return query, runtime, env, [*messages, message], extra_args


def test_sql_tool_advertised_only_when_source_required_sink_present():
    runtime, _, messages, extra_args = _bootstrap(_sr_llm())
    try:
        assert extra_args["dfc_context"].source_required_sinks  # SR sink indexed
        assert SQL_TOOL_NAME in runtime.functions
        assert "execute_sql_for_source_required" in get_text_content_as_str(messages[0]["content"])
    finally:
        extra_args["dfc_context"].close()

    runtime2, _, _, extra_args2 = _bootstrap(_dimension_llm())
    try:
        assert not extra_args2["dfc_context"].source_required_sinks
        assert SQL_TOOL_NAME not in runtime2.functions
    finally:
        extra_args2["dfc_context"].close()


def test_direct_sr_tool_call_is_redirected_to_sql_gateway():
    runtime, env, _, extra_args = _bootstrap(_sr_llm())
    executor = AgentDojoDFCToolsExecutor()
    try:
        messages = [
            ChatUserMessage(role="user", content=[text_content_block_from_string("task")]),
            _assistant("send_money", {"recipient": "DE123", "amount": "100"}, "c1"),
        ]
        _, _, _, messages, _ = executor.query("task", runtime, env, messages, extra_args)
        error = messages[-1]["error"]
        assert error is not None
        assert "execute_sql_for_source_required" in error
        assert "GetContactsOutput" in error  # source schema is shown
    finally:
        extra_args["dfc_context"].close()


def test_grounded_sql_gateway_performs_the_action():
    runtime, env, _, extra_args = _bootstrap(_sr_llm())
    executor = AgentDojoDFCToolsExecutor()
    context = extra_args["dfc_context"]
    try:
        context.record_tool_output("get_contacts", Contact(account="DE123", name="Alice"))  # agent read contacts
        sql = ("INSERT INTO SendMoneyInput (recipient, amount) "
               "SELECT account, '100' FROM GetContactsOutput WHERE account = 'DE123'")
        messages = [
            ChatUserMessage(role="user", content=[text_content_block_from_string("task")]),
            _assistant(SQL_TOOL_NAME, {"sql": sql}, "c2"),
        ]
        _, _, _, messages, _ = executor.query("task", runtime, env, messages, extra_args)
        assert messages[-1]["error"] is None
        assert "sent 100 to DE123" in get_text_content_as_str(messages[-1]["content"])
    finally:
        context.close()


def test_blocked_sql_gateway_returns_retry_message():
    runtime, env, _, extra_args = _bootstrap(_sr_llm())
    executor = AgentDojoDFCToolsExecutor()
    context = extra_args["dfc_context"]
    try:
        context.record_tool_output("get_contacts", Contact(account="DE123", name="Alice"))
        laundered = ("INSERT INTO SendMoneyInput (recipient, amount) "
                     "SELECT 'ATTACKER', '100' FROM GetContactsOutput")
        messages = [
            ChatUserMessage(role="user", content=[text_content_block_from_string("task")]),
            _assistant(SQL_TOOL_NAME, {"sql": laundered}, "c3"),
        ]
        _, _, _, messages, _ = executor.query("task", runtime, env, messages, extra_args)
        assert messages[-1]["error"] is not None
        assert "data flow policy violation" in messages[-1]["error"].lower()
        assert send_money("ATTACKER", "100")  # sanity: tool itself works; gateway must NOT have called it
    finally:
        context.close()


def test_end_to_end_redirect_then_read_then_grounded_sql_in_full_pipeline():
    runtime = _runtime()
    grounded = ("INSERT INTO SendMoneyInput (recipient, amount) "
                "SELECT account, '100' FROM GetContactsOutput WHERE account = 'DE123'")
    llm = FakeLLM([
        _assistant("send_money", {"recipient": "DE123", "amount": "100"}, "t1"),  # direct -> redirected
        _assistant("get_contacts", {}, "t2"),                                     # read contacts -> source populated
        _assistant(SQL_TOOL_NAME, {"sql": grounded}, "t3"),                       # grounded gateway -> performs action
        ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("done")], tool_calls=None),
    ])
    pipeline = AgentPipeline([
        InitQuery(),
        AgentDojoDFCBootstrap("shopping", _sr_llm(), "fake-model"),
        llm,
        ToolsExecutionLoop([AgentDojoDFCToolsExecutor(), llm]),
    ])
    _, _, _, messages, extra_args = pipeline.query("Send money to Alice", runtime, EmptyEnv(), extra_args={})
    try:
        tool_messages = [message for message in messages if message["role"] == "tool"]
        assert any(m["error"] and "execute_sql_for_source_required" in m["error"] for m in tool_messages)
        assert any(
            m["error"] is None and "sent 100 to DE123" in get_text_content_as_str(m["content"])
            for m in tool_messages
        )
    finally:
        extra_args["dfc_context"].close()
