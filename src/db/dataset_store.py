"""MongoDB storage for bounded tabular datasets."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.db.mongo_client import db

datasets = db["datasets"]
dataset_batches = db["dataset_batches"]


async def initialize_dataset_store() -> None:
    await datasets.create_index("dataset_id", unique=True)
    await datasets.create_index([("session_id", 1), ("created_at", -1)])
    await dataset_batches.create_index([("dataset_id", 1), ("batch_index", 1)], unique=True)


async def save_dataframe(
    frame: pd.DataFrame,
    *,
    session_id: str,
    filename: str,
    description: str,
) -> dict:
    dataset_id = str(uuid4())
    columns = [str(column) for column in frame.columns]
    normalized = frame.copy()
    normalized.columns = columns
    await datasets.insert_one(
        {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "filename": filename,
            "description": description,
            "columns": columns,
            "dtypes": {column: str(dtype) for column, dtype in normalized.dtypes.items()},
            "row_count": len(normalized),
            "created_at": datetime.now(timezone.utc),
        }
    )
    documents = []
    for batch_index, start in enumerate(range(0, len(normalized), 500)):
        batch = normalized.iloc[start : start + 500]
        records = json.loads(batch.to_json(orient="records", date_format="iso"))
        documents.append({"dataset_id": dataset_id, "batch_index": batch_index, "rows": records})
    if documents:
        await dataset_batches.insert_many(documents, ordered=True)
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "description": description,
        "columns": columns,
        "row_count": len(normalized),
    }


async def load_dataframe(dataset_id: str, session_id: str) -> tuple[pd.DataFrame, dict]:
    metadata = await datasets.find_one(
        {"dataset_id": dataset_id, "session_id": session_id}, {"_id": 0}
    )
    if metadata is None:
        raise ValueError("Dataset not found in this workspace")
    cursor = dataset_batches.find({"dataset_id": dataset_id}, {"_id": 0}).sort("batch_index", 1)
    batches = await cursor.to_list(length=1000)
    rows = [row for batch in batches for row in batch["rows"]]
    return pd.DataFrame(rows, columns=metadata["columns"]), metadata


async def list_datasets(session_id: str) -> list[dict]:
    cursor = datasets.find({"session_id": session_id}, {"_id": 0}).sort("created_at", -1)
    return await cursor.to_list(length=100)
