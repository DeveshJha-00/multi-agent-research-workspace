"""Async MongoDB checkpoint saver for durable LangGraph execution."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from bson import Binary
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from src.db.mongo_client import db


class MongoCheckpointSaver(BaseCheckpointSaver):
    """Persist complete checkpoints and pending writes in the application's MongoDB."""

    def __init__(self) -> None:
        super().__init__()
        self.checkpoints = db["langgraph_checkpoints"]
        self.writes = db["langgraph_writes"]

    async def initialize(self) -> None:
        await self.checkpoints.create_index(
            [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)], unique=True
        )
        await self.writes.create_index(
            [
                ("thread_id", 1),
                ("checkpoint_ns", 1),
                ("checkpoint_id", 1),
                ("task_id", 1),
                ("idx", 1),
            ],
            unique=True,
        )

    def _typed(self, value: Any) -> dict[str, Any]:
        type_name, payload = self.serde.dumps_typed(value)
        return {"type": type_name, "payload": Binary(payload)}

    def _load(self, value: dict[str, Any]) -> Any:
        return self.serde.loads_typed((value["type"], bytes(value["payload"])))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        query: dict[str, Any] = {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
        }
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id
        document = await self.checkpoints.find_one(query, sort=[("checkpoint_id", -1)])
        if document is None:
            return None

        checkpoint = self._load(document["checkpoint"])
        metadata = self._load(document["metadata"])
        write_query = {
            "thread_id": document["thread_id"],
            "checkpoint_ns": document["checkpoint_ns"],
            "checkpoint_id": document["checkpoint_id"],
        }
        pending_writes = []
        async for write in self.writes.find(write_query).sort([("task_id", 1), ("idx", 1)]):
            pending_writes.append((write["task_id"], write["channel"], self._load(write["value"])))

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": document["thread_id"],
                "checkpoint_ns": document["checkpoint_ns"],
                "checkpoint_id": document["checkpoint_id"],
            }
        }
        parent_config = None
        if document.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": document["thread_id"],
                    "checkpoint_ns": document["checkpoint_ns"],
                    "checkpoint_id": document["parent_checkpoint_id"],
                }
            }
        return CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query: dict[str, Any] = {}
        if config:
            configurable = config["configurable"]
            query["thread_id"] = configurable["thread_id"]
            query["checkpoint_ns"] = configurable.get("checkpoint_ns", "")
            checkpoint_id = get_checkpoint_id(config)
            if checkpoint_id:
                query["checkpoint_id"] = checkpoint_id
        if filter:
            for key, value in filter.items():
                query[f"metadata_search.{key}"] = value
        if before and (before_id := get_checkpoint_id(before)):
            query["checkpoint_id"] = {"$lt": before_id}

        cursor = self.checkpoints.find(query).sort("checkpoint_id", -1)
        if limit:
            cursor = cursor.limit(limit)
        async for document in cursor:
            exact_config: RunnableConfig = {
                "configurable": {
                    "thread_id": document["thread_id"],
                    "checkpoint_ns": document["checkpoint_ns"],
                    "checkpoint_id": document["checkpoint_id"],
                }
            }
            item = await self.aget_tuple(exact_config)
            if item is not None:
                yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        merged_metadata = get_checkpoint_metadata(config, metadata)
        await self.checkpoints.replace_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": configurable.get("checkpoint_id"),
                "checkpoint": self._typed(checkpoint),
                "metadata": self._typed(merged_metadata),
                "metadata_search": dict(merged_metadata),
            },
            upsert=True,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        for position, (channel, value) in enumerate(writes):
            index = WRITES_IDX_MAP.get(channel, position)
            query = {
                "thread_id": configurable["thread_id"],
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": configurable["checkpoint_id"],
                "task_id": task_id,
                "idx": index,
            }
            update = (
                {
                    "$set": {
                        **query,
                        "task_path": task_path,
                        "channel": channel,
                        "value": self._typed(value),
                    }
                }
                if index < 0
                else {
                    "$setOnInsert": {
                        **query,
                        "task_path": task_path,
                        "channel": channel,
                        "value": self._typed(value),
                    }
                }
            )
            await self.writes.update_one(
                query,
                update,
                upsert=True,
            )

    async def adelete_thread(self, thread_id: str) -> None:
        await self.checkpoints.delete_many({"thread_id": thread_id})
        await self.writes.delete_many({"thread_id": thread_id})


checkpoint_saver = MongoCheckpointSaver()


async def initialize_checkpoint_store() -> None:
    await checkpoint_saver.initialize()
