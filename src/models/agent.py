"""Shared schemas for multi-agent plans, evidence, and results."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

AgentName = Literal[
    "document_investigator",
    "web_researcher",
    "data_analyst",
    "repository_analyst",
]


class AgentTask(BaseModel):
    agent: AgentName
    instruction: str = Field(min_length=5, max_length=2000)
    rationale: str = Field(min_length=3, max_length=500)


class ResearchPlan(BaseModel):
    objective: str
    tasks: list[AgentTask] = Field(min_length=1, max_length=8)


class AgentResult(BaseModel):
    agent: str
    instruction: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    error: str | None = None


class FollowUpTask(BaseModel):
    agent: AgentName
    instruction: str
    rationale: str


class Critique(BaseModel):
    approved: bool
    coverage_score: float = Field(ge=0.0, le=1.0)
    problems: list[str] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list, max_length=3)


class ArtifactRecord(BaseModel):
    artifact_id: str
    name: str
    media_type: str


class OrchestrationState(TypedDict, total=False):
    task_id: str
    session_id: str
    objective: str
    available_data: list[str]
    plan: list[AgentTask]
    worker_results: Annotated[list[AgentResult], operator.add]
    critique: Critique
    revision_count: int
    final_answer: str
    artifacts: list[ArtifactRecord]


class WorkerState(TypedDict):
    task_id: str
    session_id: str
    objective: str
    task: AgentTask
    worker_results: list[AgentResult]
