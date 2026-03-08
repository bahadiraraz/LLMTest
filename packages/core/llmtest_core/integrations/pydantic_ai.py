"""
Pydantic AI integration for llmtest.

Wraps Pydantic AI agents so their outputs can be tested
with llmtest assertions (contains, cost_under, tool_called, etc.).

Install: pip install pydantic-ai

Usage:
    from pydantic_ai import Agent
    from llmtest.integrations.pydantic_ai import PydanticAIAdapter
    from llmtest import expect

    agent = Agent('openai:gpt-5-mini', system_prompt='Be concise.')
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("What is the capital of France?")
    assert expect.contains("Paris").check(output).passed
    assert expect.latency_under(5000).check(output).passed
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from llmtest_core.models import LLMOutput

if TYPE_CHECKING:
    from pydantic_ai import Agent


class PydanticAIAdapter:
    """
    Wraps a Pydantic AI Agent for testing with llmtest assertions.

    Runs the agent and converts the result to LLMOutput,
    which is compatible with all llmtest assertions.

    Args:
        agent: A pydantic_ai.Agent instance.
        deps: Default dependencies to pass to the agent.
        model: Override the agent's default model for testing.
        model_settings: Default ModelSettings for all runs.

    Example:
        from pydantic_ai import Agent
        from llmtest.integrations.pydantic_ai import PydanticAIAdapter
        from llmtest import expect

        agent = Agent('openai:gpt-5-mini', system_prompt='Be concise.')
        adapter = PydanticAIAdapter(agent)

        # Synchronous
        output = adapter.run_sync("Capital of France?")
        assert expect.contains("Paris").check(output).passed
        assert output.latency_ms < 5000
        assert output.input_tokens > 0

        # With deps
        adapter = PydanticAIAdapter(agent, deps=MyDeps())
        output = adapter.run_sync("Do something", deps=override_deps)
    """

    def __init__(
        self,
        agent: Agent,
        deps: Any = None,
        model: str | None = None,
        model_settings: Any = None,
    ):
        self.agent = agent
        self.deps = deps
        self.model = model
        self.model_settings = model_settings

    def run_sync(
        self,
        prompt: str,
        *,
        deps: Any = None,
        model: str | None = None,
        message_history: list | None = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        retry_if: Any = None,
        **kwargs: Any,
    ) -> LLMOutput:
        """
        Run the agent synchronously and return LLMOutput.

        Args:
            prompt: User prompt to send to the agent.
            deps: Dependencies (overrides adapter-level deps).
            model: Model override for this run.
            message_history: Previous messages for multi-turn.
            model_settings: Model settings for this run.
            usage_limits: Usage limits (pydantic_ai.UsageLimits).
            retries: Number of retries if retry_if returns True.
            retry_delay: Seconds to wait between retries.
            retry_if: Callable(LLMOutput) -> bool. Retry if True.
            **kwargs: Extra kwargs passed to agent.run_sync().

        Returns:
            LLMOutput with content, tokens, latency, tool_calls, etc.
        """
        run_kwargs = self._build_run_kwargs(
            deps=deps,
            model=model,
            message_history=message_history,
            model_settings=model_settings,
            usage_limits=usage_limits,
            **kwargs,
        )

        should_retry = retry_if or (lambda out: out.finish_reason == "error")

        for attempt in range(retries + 1):
            if attempt > 0 and retry_delay > 0:
                time.sleep(retry_delay)

            start = time.monotonic()
            try:
                result = self.agent.run_sync(prompt, **run_kwargs)
            except Exception as e:
                output = LLMOutput(
                    content="",
                    model=self.model or str(getattr(self.agent, "model", "")),
                    latency_ms=(time.monotonic() - start) * 1000,
                    finish_reason="error",
                    raw={"error": str(e), "pydantic_ai": True},
                )
                if not should_retry(output) or attempt == retries:
                    return output
                continue

            latency_ms = (time.monotonic() - start) * 1000
            output = self._to_llm_output(result, latency_ms)

            if not should_retry(output) or attempt == retries:
                return output

        return output  # unreachable

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        model: str | None = None,
        message_history: list | None = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        retry_if: Any = None,
        **kwargs: Any,
    ) -> LLMOutput:
        """
        Run the agent asynchronously and return LLMOutput.
        Same parameters as run_sync().
        """
        import asyncio

        run_kwargs = self._build_run_kwargs(
            deps=deps,
            model=model,
            message_history=message_history,
            model_settings=model_settings,
            usage_limits=usage_limits,
            **kwargs,
        )

        should_retry = retry_if or (lambda out: out.finish_reason == "error")

        for attempt in range(retries + 1):
            if attempt > 0 and retry_delay > 0:
                await asyncio.sleep(retry_delay)

            start = time.monotonic()
            try:
                result = await self.agent.run(prompt, **run_kwargs)
            except Exception as e:
                output = LLMOutput(
                    content="",
                    model=self.model or str(getattr(self.agent, "model", "")),
                    latency_ms=(time.monotonic() - start) * 1000,
                    finish_reason="error",
                    raw={"error": str(e), "pydantic_ai": True},
                )
                if not should_retry(output) or attempt == retries:
                    return output
                continue

            latency_ms = (time.monotonic() - start) * 1000
            output = self._to_llm_output(result, latency_ms)

            if not should_retry(output) or attempt == retries:
                return output

        return output  # unreachable

    def _build_run_kwargs(
        self,
        *,
        deps,
        model,
        message_history,
        model_settings,
        usage_limits,
        **extra,
    ) -> dict[str, Any]:
        """Build kwargs dict for agent.run() / agent.run_sync()."""
        kw: dict[str, Any] = {}

        resolved_deps = deps if deps is not None else self.deps
        if resolved_deps is not None:
            kw["deps"] = resolved_deps

        resolved_model = model or self.model
        if resolved_model is not None:
            kw["model"] = resolved_model

        if message_history is not None:
            kw["message_history"] = message_history

        resolved_settings = model_settings or self.model_settings
        if resolved_settings is not None:
            kw["model_settings"] = resolved_settings

        if usage_limits is not None:
            kw["usage_limits"] = usage_limits

        kw.update(extra)
        return kw

    def _to_llm_output(self, result: Any, latency_ms: float) -> LLMOutput:
        """Convert a Pydantic AI AgentRunResult to llmtest LLMOutput."""
        from pydantic import BaseModel as PydanticBaseModel

        # --- Content ---
        raw_output = result.output
        if isinstance(raw_output, str):
            content = raw_output
        elif isinstance(raw_output, bool):
            content = str(raw_output).lower()
        elif isinstance(raw_output, PydanticBaseModel):
            content = raw_output.model_dump_json(indent=2)
        elif raw_output is not None:
            try:
                content = json.dumps(raw_output, default=str)
            except (TypeError, ValueError):
                content = str(raw_output)
        else:
            content = ""

        # --- Usage ---
        usage = result.usage()
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0

        # --- Tool calls ---
        tool_calls = self._extract_tool_calls(result.new_messages())

        # --- Model name ---
        model_name = ""
        for msg in result.new_messages():
            if hasattr(msg, "model_name") and msg.model_name:
                model_name = msg.model_name
                break

        return LLMOutput(
            content=content,
            model=model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason="stop",
            raw={
                "pydantic_ai": True,
                "requests": usage.requests,
                "tool_calls_count": usage.tool_calls,
                "total_tokens": usage.total_tokens,
                "output_type": type(raw_output).__name__,
            },
        )

    @staticmethod
    def _extract_tool_calls(messages: list) -> list[dict[str, Any]]:
        """Extract tool calls from Pydantic AI message history."""
        tool_calls: list[dict[str, Any]] = []
        for msg in messages:
            if not hasattr(msg, "parts"):
                continue
            for part in msg.parts:
                # ToolCallPart has tool_name, args, tool_call_id
                if not hasattr(part, "tool_name"):
                    continue
                # Skip non-tool-call parts
                part_kind = getattr(part, "part_kind", "")
                if part_kind not in ("tool-call", "builtin-tool-call"):
                    continue

                args: dict[str, Any] = {}
                if hasattr(part, "args_as_dict"):
                    args = part.args_as_dict()
                elif isinstance(getattr(part, "args", None), dict):
                    args = part.args
                elif isinstance(getattr(part, "args", None), str):
                    try:
                        args = json.loads(part.args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                tool_calls.append(
                    {
                        "id": getattr(part, "tool_call_id", ""),
                        "name": part.tool_name,
                        "arguments": args,
                    }
                )
        return tool_calls
