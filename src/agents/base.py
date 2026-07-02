"""Reusable bounded tool-calling agent loop."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.core.config import settings
from src.llms.openai import get_llm
from src.models.agent import AgentResult

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    task_id: str
    session_id: str
    objective: str
    instruction: str
    prior_results: list[AgentResult] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)


class ToolCallingAgent:
    """A specialist that autonomously selects from a constrained toolset."""

    name = "specialist"
    system_prompt = (
        "Use your tools to complete the assigned task and report evidence-backed findings."
    )

    def build_tools(self, context: AgentContext) -> list[BaseTool]:
        raise NotImplementedError

    async def run(self, context: AgentContext) -> AgentResult:
        tools = self.build_tools(context)
        tool_map = {tool.name: tool for tool in tools}
        model = get_llm().bind_tools(tools)
        prior = "\n".join(
            f"- {item.agent}: {item.summary[:1500]}" for item in context.prior_results[-6:]
        )
        messages = [
            SystemMessage(
                content=(
                    f"{self.system_prompt}\n"
                    f"You are the {self.name}. Use tools when evidence is required. "
                    "Never fabricate tool results. Return a concise final summary for the supervisor."
                )
            ),
            HumanMessage(
                content=(
                    f"Overall objective: {context.objective}\n"
                    f"Your assignment: {context.instruction}\n"
                    f"Prior specialist results:\n{prior or 'None'}"
                )
            ),
        ]

        tool_calls = 0
        try:
            for _ in range(settings.agent_max_iterations):
                response = await model.ainvoke(messages)
                messages.append(response)
                if not response.tool_calls:
                    return AgentResult(
                        agent=self.name,
                        instruction=context.instruction,
                        summary=str(response.content),
                        evidence_ids=context.evidence_ids,
                        tool_calls=tool_calls,
                    )

                for call in response.tool_calls:
                    tool_calls += 1
                    tool = tool_map.get(call["name"])
                    if tool is None:
                        content = json.dumps({"error": f"Unknown tool: {call['name']}"})
                    else:
                        try:
                            output: Any = await tool.ainvoke(call.get("args", {}))
                            content = (
                                output
                                if isinstance(output, str)
                                else json.dumps(output, default=str)
                            )
                        except Exception as exc:
                            logger.exception(
                                "agent_tool_failed agent=%s tool=%s", self.name, call["name"]
                            )
                            content = json.dumps({"error": str(exc)})
                    messages.append(ToolMessage(content=content[:16000], tool_call_id=call["id"]))

            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary="The specialist reached its tool-call limit before producing a final response.",
                evidence_ids=context.evidence_ids,
                tool_calls=tool_calls,
                error="iteration_limit",
            )
        except Exception as exc:
            logger.exception("agent_failed agent=%s", self.name)
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary="The specialist failed to complete its assignment.",
                evidence_ids=context.evidence_ids,
                tool_calls=tool_calls,
                error=str(exc),
            )
