"""DFCToolsExecutor — the gate. For each tool call in the last assistant message:
  - execute_intent(sql=...)  -> routed to DFCSession (enforce + stage + drain to the real guarded tool)
  - any other tool          -> executed normally (the non-guarded reals)

Reuses the base ToolsExecutor's message-shape guards and output formatter."""

from ast import literal_eval
from collections.abc import Sequence

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, is_string_list
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string


class DFCToolsExecutor(ToolsExecutor):
    def _result(self, tc, text: str, error: str | None) -> ChatToolResultMessage:
        return ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string(text or "")],
            tool_call_id=tc.id,
            tool_call=tc,
            error=error,
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args

        session = extra_args.get("dfc_session")
        results: list[ChatToolResultMessage] = []
        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == "execute_intent":
                if session is None:
                    results.append(self._result(tool_call, "", "DFC session missing."))
                    continue
                text, error = session.handle_intent(str(tool_call.args.get("sql", "")), env)
                results.append(self._result(tool_call, text, error))
                continue

            # ---- normal (non-guarded) tool call ----
            if tool_call.function not in (t.name for t in runtime.functions.values()):
                results.append(self._result(tool_call, "", f"Invalid tool {tool_call.function} provided."))
                continue
            for k, v in tool_call.args.items():
                if isinstance(v, str) and is_string_list(v):
                    tool_call.args[k] = literal_eval(v)
            result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            results.append(self._result(tool_call, self.output_formatter(result), error))

        return query, runtime, env, [*messages, *results], extra_args
