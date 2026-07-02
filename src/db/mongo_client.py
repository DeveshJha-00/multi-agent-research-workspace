"""MongoDB client lifecycle and readiness helpers."""

from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import settings

client = AsyncIOMotorClient(
    settings.mongodb_url,
    serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
    uuidRepresentation="standard",
)
db = client[settings.mongodb_db_name]


async def initialize_mongodb() -> None:
    await client.admin.command("ping")
    await db["chat_history"].create_index([("session_id", 1), ("_id", -1)])


async def mongodb_ready() -> bool:
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False


def close_mongodb() -> None:
    client.close()
