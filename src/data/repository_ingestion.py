"""Safe, bounded ingestion of ZIP source repositories."""

import asyncio
import hashlib
import stat
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile

from src.core.config import settings
from src.db.repository_store import save_repository

IGNORED_PARTS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "venv",
}
TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".dockerfile",
    ".env.example",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".md",
    ".php",
    ".properties",
    ".proto",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".svelte",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    ".env.example",
    "dockerfile",
    "gemfile",
    "license",
    "makefile",
    "procfile",
    "readme",
    "requirements.txt",
}
LANGUAGES = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
}


def _safe_path(raw_name: str) -> str | None:
    normalized = raw_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.name or ":" in path.parts[0]:
        return None
    parts = [part for part in path.parts if part not in {"", "."}]
    if any(part.casefold() in IGNORED_PARTS for part in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def _is_text_file(path: str) -> bool:
    file_path = PurePosixPath(path)
    name = file_path.name.casefold()
    return file_path.suffix.casefold() in TEXT_EXTENSIONS or name in TEXT_FILENAMES


def _parse_repository(content: bytes) -> tuple[list[dict], dict[str, int]]:
    try:
        archive = ZipFile(BytesIO(content))
    except BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid ZIP archive") from exc

    files = []
    languages: Counter[str] = Counter()
    total_bytes = 0
    seen_paths = set()
    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                continue
            path = _safe_path(member.filename)
            if path is None or not _is_text_file(path):
                continue
            if path in seen_paths:
                continue
            if member.file_size > settings.max_repository_file_bytes:
                continue
            if len(files) >= settings.max_repository_files:
                raise ValueError(
                    "Repository contains more supported files than the configured limit"
                )
            with archive.open(member) as source:
                raw = source.read(settings.max_repository_file_bytes + 1)
            if len(raw) > settings.max_repository_file_bytes:
                continue
            if b"\x00" in raw[:4096]:
                continue
            total_bytes += len(raw)
            if total_bytes > settings.max_repository_total_bytes:
                raise ValueError("Repository text exceeds the configured extracted-size limit")
            text = raw.decode("utf-8", errors="replace")
            extension = PurePosixPath(path).suffix.casefold()
            language = LANGUAGES.get(extension)
            if language:
                languages[language] += 1
            files.append({"path": path, "content": text, "size": len(raw), "extension": extension})
            seen_paths.add(path)
    if not files:
        raise ValueError("ZIP contains no supported text or source files")
    return files, dict(languages.most_common())


async def ingest_repository(*, file: UploadFile, session_id: str, description: str = "") -> dict:
    filename = Path(file.filename or "").name
    if Path(filename).suffix.casefold() != ".zip":
        raise HTTPException(status_code=400, detail="Only ZIP repository uploads are supported")
    content = await file.read(settings.max_repository_upload_bytes + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded repository is empty")
    if len(content) > settings.max_repository_upload_bytes:
        raise HTTPException(
            status_code=413, detail="Repository ZIP exceeds the configured upload limit"
        )
    try:
        files, languages = await asyncio.to_thread(_parse_repository, content)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await save_repository(
        session_id=session_id,
        filename=filename,
        description=description.strip(),
        files=files,
        languages=languages,
        content_hash=hashlib.sha256(content).hexdigest(),
    )
