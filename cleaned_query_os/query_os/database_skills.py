from __future__ import annotations

from pathlib import Path


def load_database_skills(db_id: str) -> str:
    """Load optional database-specific skills for runtime worker context."""
    safe_db_id = (db_id or "").strip()
    if not safe_db_id or "/" in safe_db_id or "\\" in safe_db_id or safe_db_id in {".", ".."}:
        return ""
    root = Path(__file__).resolve().parents[1] / "database_skills"
    target = root / safe_db_id / "skills.md"
    if not target.exists() or not target.is_file():
        return ""
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
