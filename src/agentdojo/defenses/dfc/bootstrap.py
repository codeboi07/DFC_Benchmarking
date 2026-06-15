"""DFCBootstrap — a pipeline element placed before the LLM that, per task:
  1. captures the real suite tools into a separate runtime (held by the DFCSession),
  2. restricts the ADVERTISED tools to (non-guarded reals) + execute_intent,
  3. builds the DFCSession (DuckDB mirror of `trusted` + staging + registered policies),
  4. injects the relational-schema block into the system message,
  5. stashes the session in extra_args for DFCToolsExecutor.

State is shared via extra_args (the camel pattern), not globals (the progent pattern)."""

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.defenses.dfc.gateway import execute_intent
from agentdojo.defenses.dfc.materialize import GUARDED_TOOLS, URL_GUARDED_TOOLS
from agentdojo.defenses.dfc.session import DFCSession
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime, make_function
from agentdojo.types import ChatMessage, text_content_block_from_string


def _augment_system(messages: Sequence[ChatMessage], doc: str) -> list[ChatMessage]:
    msgs = list(messages)
    block = text_content_block_from_string("\n\n" + doc)
    if msgs and msgs[0]["role"] == "system":
        sys = dict(msgs[0])
        sys["content"] = [*(sys.get("content") or []), block]
        msgs[0] = sys  # type: ignore[assignment]
    else:
        msgs.insert(0, {"role": "system", "content": [block]})  # type: ignore[arg-type]
    return msgs


class DFCBootstrap(BasePipelineElement):
    def __init__(self, suite_name: str | None, gate_urls: bool = False) -> None:
        self.suite_name = suite_name
        self.gate_urls = gate_urls

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # 1. capture the full real-tool runtime (incl. guarded tools, used by the interpreter)
        real_runtime = FunctionsRuntime(list(runtime.functions.values()))

        # 2. build the per-task DFC session (trusted + staging + policies)
        session = DFCSession(env, real_runtime, query, self.suite_name, gate_urls=self.gate_urls)

        # 3. advertise only non-guarded reals + execute_intent (this is what the LLM sees)
        hidden = GUARDED_TOOLS | (URL_GUARDED_TOOLS if self.gate_urls else set())
        advertised = {name: f for name, f in runtime.functions.items() if name not in hidden}
        ei = make_function(execute_intent)
        advertised[ei.name] = ei
        runtime.functions = advertised

        # 4. inject the relational-schema block; 5. share the session
        messages = _augment_system(messages, session.schema_doc())
        extra_args = {**extra_args, "dfc_session": session}
        return query, runtime, env, messages, extra_args
