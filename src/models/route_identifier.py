"""Structured query route returned by the classifier."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteIdentifier(BaseModel):
    route: Literal["index", "general", "search"]
    reason: str = Field(min_length=1, max_length=2000)
