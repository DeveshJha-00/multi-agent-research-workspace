"""MongoDB storage and bounded inspection for uploaded source repositories."""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import NAMESPACE_URL, uuid5

from pymongo import UpdateOne

from src.core.config import settings
from src.db.mongo_client import db

repositories = db["repositories"]
repository_files = db["repository_files"]


async def initialize_repository_store() -> None:
    await repositories.create_index("repository_id", unique=True)
    await repositories.create_index([("session_id", 1), ("created_at", -1)])
    await repositories.create_index(
        [("session_id", 1), ("content_hash", 1)],
        unique=True,
        partialFilterExpression={"content_hash": {"$type": "string"}},
    )
    await repository_files.create_index([("repository_id", 1), ("path", 1)], unique=True)


async def save_repository(
    *,
    session_id: str,
    filename: str,
    description: str,
    files: list[dict],
    languages: dict[str, int],
    content_hash: str,
) -> dict:
    repository_id = str(uuid5(NAMESPACE_URL, f"repository:{session_id}:{content_hash}"))
    now = datetime.now(timezone.utc)
    total_bytes = sum(item["size"] for item in files)
    metadata = {
        "repository_id": repository_id,
        "session_id": session_id,
        "filename": filename,
        "description": description,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "languages": languages,
        "content_hash": content_hash,
        "status": "ingesting",
        "created_at": now,
    }
    write = await repositories.update_one(
        {"repository_id": repository_id}, {"$setOnInsert": metadata}, upsert=True
    )
    reused = write.upserted_id is None
    operations = [
        UpdateOne(
            {"repository_id": repository_id, "path": item["path"]},
            {
                "$setOnInsert": {
                    "repository_id": repository_id,
                    "path": item["path"],
                    "content": item["content"],
                    "size": item["size"],
                    "extension": item["extension"],
                }
            },
            upsert=True,
        )
        for item in files
    ]
    try:
        for start in range(0, len(operations), 250):
            await repository_files.bulk_write(operations[start : start + 250], ordered=False)
        await repositories.update_one(
            {"repository_id": repository_id},
            {"$set": {"status": "ready", "updated_at": datetime.now(timezone.utc)}},
        )
    except Exception:
        if not reused:
            await repositories.delete_one({"repository_id": repository_id})
            await repository_files.delete_many({"repository_id": repository_id})
        raise
    stored = await repositories.find_one({"repository_id": repository_id})
    if stored is None:
        raise RuntimeError("Repository metadata disappeared during ingestion")
    return {
        key: value
        for key, value in {**stored, "reused": reused}.items()
        if key not in {"_id", "session_id", "content_hash", "created_at", "updated_at", "status"}
    }


async def get_repository(repository_id: str, session_id: str) -> dict | None:
    return await repositories.find_one(
        {"repository_id": repository_id, "session_id": session_id},
        {"_id": 0, "session_id": 0, "content_hash": 0},
    )


async def list_repositories(session_id: str) -> list[dict]:
    cursor = repositories.find(
        {
            "session_id": session_id,
            "$or": [{"status": "ready"}, {"status": {"$exists": False}}],
        },
        {"_id": 0, "session_id": 0, "content_hash": 0},
    ).sort("created_at", -1)
    return await cursor.to_list(length=100)


async def list_repository_files(
    repository_id: str,
    session_id: str,
    *,
    prefix: str = "",
    limit: int = 500,
) -> list[dict]:
    if await get_repository(repository_id, session_id) is None:
        raise ValueError("Repository not found in this workspace")
    normalized_prefix = _normalize_path(prefix) if prefix else ""
    query: dict = {"repository_id": repository_id}
    if normalized_prefix:
        query["path"] = {"$regex": f"^{_escape_regex(normalized_prefix)}"}
    cursor = (
        repository_files.find(query, {"_id": 0, "content": 0})
        .sort("path", 1)
        .limit(max(1, min(limit, 1000)))
    )
    return await cursor.to_list(length=max(1, min(limit, 1000)))


async def read_repository_file(
    repository_id: str,
    session_id: str,
    path: str,
    *,
    start_line: int = 1,
    end_line: int = 250,
) -> dict:
    if await get_repository(repository_id, session_id) is None:
        raise ValueError("Repository not found in this workspace")
    normalized_path = _normalize_path(path)
    document = await repository_files.find_one(
        {"repository_id": repository_id, "path": normalized_path}, {"_id": 0}
    )
    if document is None:
        raise ValueError("Repository file not found")
    lines = document["content"].splitlines()
    if not lines:
        return {
            "path": normalized_path,
            "start_line": 0,
            "end_line": 0,
            "total_lines": 0,
            "content": "",
        }
    first = min(max(1, start_line), len(lines))
    last = max(first, min(end_line, first + 499, len(lines)))
    selected = [f"{number}: {lines[number - 1]}" for number in range(first, last + 1)]
    return {
        "path": normalized_path,
        "start_line": first,
        "end_line": last,
        "total_lines": len(lines),
        "content": "\n".join(selected),
    }


async def search_repository_code(
    repository_id: str,
    session_id: str,
    query: str,
    *,
    path_prefix: str = "",
) -> list[dict]:
    if await get_repository(repository_id, session_id) is None:
        raise ValueError("Repository not found in this workspace")
    needle = query.strip().casefold()
    if len(needle) < 2:
        raise ValueError("Search query must contain at least two characters")
    mongo_query: dict = {"repository_id": repository_id}
    if path_prefix:
        normalized = _normalize_path(path_prefix)
        mongo_query["path"] = {"$regex": f"^{_escape_regex(normalized)}"}
    cursor = repository_files.find(mongo_query, {"_id": 0}).sort("path", 1).limit(1000)
    matches = []
    async for document in cursor:
        for line_number, line in enumerate(document["content"].splitlines(), start=1):
            if needle in line.casefold():
                matches.append(
                    {
                        "path": document["path"],
                        "line": line_number,
                        "excerpt": line.strip()[:500],
                    }
                )
                if len(matches) >= settings.repository_search_max_matches:
                    return matches
    return matches


def _escape_regex(value: str) -> str:
    """Escape a literal MongoDB regex prefix without enabling regex input."""
    special = "\\.^$*+?{}[]|()"
    return "".join(f"\\{character}" if character in special else character for character in value)


def _normalize_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Invalid repository path")
    return path.as_posix()
