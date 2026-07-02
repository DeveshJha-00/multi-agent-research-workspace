"""Bounded MongoDB-backed chat history."""

from datetime import datetime, timezone

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from src.core.config import settings
from src.db.mongo_client import db

collection = db["chat_history"]


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def add_message(self, message: BaseMessage) -> None:
        await self.add_messages([message])

    async def add_messages(self, messages: list[BaseMessage]) -> None:
        if not messages:
            return
        created_at = datetime.now(timezone.utc)
        await collection.insert_many(
            [
                {
                    "session_id": self.session_id,
                    "type": message.type,
                    "content": message.content,
                    "additional_kwargs": message.additional_kwargs,
                    "created_at": created_at,
                }
                for message in messages
            ],
            ordered=True,
        )

    async def get_messages(self) -> list[BaseMessage]:
        cursor = (
            collection.find({"session_id": self.session_id})
            .sort("_id", -1)
            .limit(settings.max_history_messages)
        )
        docs = list(reversed(await cursor.to_list(length=settings.max_history_messages)))
        return messages_from_dict(
            [
                {
                    "type": item["type"],
                    "data": {
                        "content": item["content"],
                        "additional_kwargs": item.get("additional_kwargs", {}),
                    },
                }
                for item in docs
            ]
        )

    async def clear(self) -> None:
        await collection.delete_many({"session_id": self.session_id})


class ChatHistory:
    @classmethod
    def get_session_history(
        cls, session_id: str, config: dict | None = None
    ) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(session_id)
