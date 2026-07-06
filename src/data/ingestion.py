"""Validated CSV, JSON, and Excel ingestion."""

import asyncio
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from src.core.config import settings
from src.db.dataset_store import save_dataframe

SUPPORTED_DATASETS = {".csv", ".json", ".xlsx"}


def _parse_dataset(content: bytes, extension: str) -> pd.DataFrame:
    stream = BytesIO(content)
    if extension == ".csv":
        return pd.read_csv(stream)
    if extension == ".json":
        try:
            return pd.read_json(stream)
        except ValueError:
            stream.seek(0)
            return pd.read_json(stream, lines=True)
    return pd.read_excel(stream, engine="openpyxl")


async def ingest_dataset(*, file: UploadFile, session_id: str, description: str = "") -> dict:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DATASETS:
        raise HTTPException(status_code=400, detail="Only CSV, JSON, and XLSX are supported")
    content = await file.read(settings.max_dataset_upload_bytes + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty")
    if len(content) > settings.max_dataset_upload_bytes:
        raise HTTPException(status_code=413, detail="Dataset exceeds configured upload limit")
    try:
        frame = await asyncio.to_thread(_parse_dataset, content, extension)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to parse dataset: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=422, detail="Dataset contains no rows")
    if len(frame) > settings.max_dataset_rows:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset has {len(frame)} rows; maximum is {settings.max_dataset_rows}",
        )
    if len(frame.columns) > settings.max_dataset_columns:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset has {len(frame.columns)} columns; maximum is {settings.max_dataset_columns}",
        )
    return await save_dataframe(
        frame,
        session_id=session_id,
        filename=filename,
        description=description.strip(),
    )
