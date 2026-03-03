from __future__ import annotations

"""
FastAPI application entrypoint for Notemaster notes backend.

Provides a REST API for:
- Creating, reading, updating, deleting notes
- Searching notes (FTS5 if available; LIKE fallback otherwise)
- Filtering notes by tag name
- Managing tags attached to notes (via note create/update payload)

The frontend is expected to call these endpoints with JSON bodies and receive JSON responses.
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# -----------------------------
# Configuration
# -----------------------------


def _utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-ish string with Z suffix."""
    # Matches init_db.py style (strftime with Z). We keep a consistent format.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _get_db_path() -> str:
    """
    Resolve SQLite DB file path.

    Uses SQLITE_DB if provided; otherwise defaults to a local file (myapp.db) in the backend container.
    """
    env_path = os.getenv("SQLITE_DB")
    if env_path and isinstance(env_path, str) and env_path.strip():
        return env_path
    # Default aligns with database container init_db.py DB_NAME
    return os.path.join(os.getcwd(), "myapp.db")


DB_PATH = _get_db_path()

# -----------------------------
# FastAPI app + OpenAPI metadata
# -----------------------------

openapi_tags = [
    {"name": "Health", "description": "Service health and diagnostics."},
    {"name": "Notes", "description": "Create, read, update, delete, search, and filter notes."},
]

app = FastAPI(
    title="Notemaster Notes API",
    description=(
        "Backend API for a notes app with optional tags. "
        "Supports CRUD, search, and tag filtering. "
        "Search uses SQLite FTS5 when available, otherwise falls back to LIKE queries."
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
)

# CORS for frontend usage (env-agnostic; safe for template/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic models
# -----------------------------


class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Note title (1-200 chars).")
    content: str = Field(..., min_length=1, max_length=20000, description="Note content/body.")
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tag names (unique, case-insensitive).",
        examples=[["work", "inbox"]],
    )

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("tags must be a list of strings")
        cleaned: List[str] = []
        seen = set()
        for item in v:
            if not isinstance(item, str):
                raise ValueError("each tag must be a string")
            name = item.strip()
            if not name:
                continue
            # normalize to lower for uniqueness; keep stored name lower for consistency
            norm = name.lower()
            if len(norm) > 50:
                raise ValueError("tag name too long (max 50 chars)")
            if norm in seen:
                continue
            seen.add(norm)
            cleaned.append(norm)
        return cleaned


class NoteCreate(NoteBase):
    pass


class NoteUpdate(NoteBase):
    pass


class NoteOut(BaseModel):
    id: str = Field(..., description="Note ID.")
    title: str = Field(..., description="Note title.")
    content: str = Field(..., description="Note content.")
    created_at: str = Field(..., description="Creation timestamp (UTC).")
    updated_at: str = Field(..., description="Last update timestamp (UTC).")
    tags: List[str] = Field(default_factory=list, description="List of tag names attached to this note.")


class DeleteResult(BaseModel):
    deleted: bool = Field(..., description="Whether the note was deleted.")
    id: str = Field(..., description="ID of the deleted note.")


# -----------------------------
# DB helpers
# -----------------------------


@contextmanager
def _db() -> sqlite3.Connection:
    """Context manager yielding a SQLite connection with useful pragmas enabled."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
    finally:
        try:
            conn.close()  # type: ignore[misc]
        except Exception:
            pass


def _fts_available(conn: sqlite3.Connection) -> bool:
    """Return True if the optional notes_fts FTS5 table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='notes_fts' LIMIT 1"
    ).fetchone()
    return row is not None


def _fetch_note_tags(conn: sqlite3.Connection, note_id: str) -> List[str]:
    """Fetch tags for a note as a list of tag names."""
    rows = conn.execute(
        """
        SELECT t.name
        FROM note_tags nt
        JOIN tags t ON t.id = nt.tag_id
        WHERE nt.note_id = ?
        ORDER BY t.name ASC
        """,
        (note_id,),
    ).fetchall()
    return [r["name"] for r in rows]


def _row_to_note_out(conn: sqlite3.Connection, row: sqlite3.Row) -> NoteOut:
    """Convert a notes table row to API NoteOut including tags."""
    return NoteOut(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        tags=_fetch_note_tags(conn, row["id"]),
    )


def _ensure_tag_ids(conn: sqlite3.Connection, tag_names: Sequence[str]) -> Dict[str, str]:
    """
    Ensure tags exist for each name; return mapping {name: tag_id}.

    Tag names are assumed already normalized (lowercase, trimmed).
    """
    mapping: Dict[str, str] = {}
    for name in tag_names:
        existing = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if existing:
            mapping[name] = existing["id"]
            continue
        tag_id = f"tag-{uuid.uuid4().hex}"
        conn.execute(
            "INSERT INTO tags (id, name) VALUES (?, ?)",
            (tag_id, name),
        )
        mapping[name] = tag_id
    return mapping


def _set_note_tags(conn: sqlite3.Connection, note_id: str, tag_names: Optional[Sequence[str]]) -> None:
    """Replace note's tags with the provided set (None => no change; [] => clear)."""
    if tag_names is None:
        return

    # Clear existing relationships
    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))

    if len(tag_names) == 0:
        return

    tag_ids = _ensure_tag_ids(conn, tag_names)
    now = _utc_now_iso()
    for name, tag_id in tag_ids.items():
        conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id, created_at) VALUES (?, ?, ?)",
            (note_id, tag_id, now),
        )


# -----------------------------
# Routes
# -----------------------------


@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    description="Simple health check endpoint.",
    operation_id="health_check",
)
def health_check() -> Dict[str, str]:
    # PUBLIC_INTERFACE
    """Health check endpoint.

    Returns:
        JSON with a simple status message.
    """
    return {"message": "Healthy"}


@app.get(
    "/notes",
    response_model=List[NoteOut],
    tags=["Notes"],
    summary="List notes",
    description=(
        "List notes ordered by updated_at desc. "
        "Optional query param `q` performs full-text search (FTS5 if available; LIKE fallback). "
        "Optional query param `tag` filters by tag name."
    ),
    operation_id="list_notes",
)
def list_notes(
    q: Optional[str] = Query(default=None, description="Optional search query."),
    tag: Optional[str] = Query(default=None, description="Optional tag name filter (case-insensitive)."),
) -> List[NoteOut]:
    # PUBLIC_INTERFACE
    """List notes with optional search and tag filtering."""
    q_norm = q.strip() if isinstance(q, str) else None
    tag_norm = tag.strip().lower() if isinstance(tag, str) else None
    if tag_norm == "":
        tag_norm = None

    with _db() as conn:
        params: List[Any] = []
        where: List[str] = []
        joins: List[str] = []

        if tag_norm:
            joins.append("JOIN note_tags nt ON nt.note_id = n.id")
            joins.append("JOIN tags t ON t.id = nt.tag_id")
            where.append("t.name = ?")
            params.append(tag_norm)

        if q_norm:
            if _fts_available(conn):
                # Use FTS5 MATCH against notes_fts.
                joins.append("JOIN notes_fts f ON f.note_id = n.id")
                where.append("notes_fts MATCH ?")
                params.append(q_norm)
            else:
                where.append("(n.title LIKE ? OR n.content LIKE ?)")
                like = f"%{q_norm}%"
                params.extend([like, like])

        sql = """
            SELECT n.id, n.title, n.content, n.created_at, n.updated_at
            FROM notes n
        """
        if joins:
            sql += "\n" + "\n".join(joins)
        if where:
            sql += "\nWHERE " + " AND ".join(where)
        sql += "\nORDER BY n.updated_at DESC"

        rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_note_out(conn, r) for r in rows]


@app.post(
    "/notes",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Create note",
    description="Create a note with optional tags.",
    operation_id="create_note",
)
def create_note(payload: NoteCreate) -> NoteOut:
    # PUBLIC_INTERFACE
    """Create a note and optionally attach tags."""
    note_id = f"note-{uuid.uuid4().hex}"
    now = _utc_now_iso()
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO notes (id, title, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (note_id, payload.title, payload.content, now, now),
        )
        _set_note_tags(conn, note_id, payload.tags)

        row = conn.execute(
            "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Failed to create note")
        return _row_to_note_out(conn, row)


@app.get(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Get note",
    description="Get a single note by ID.",
    operation_id="get_note",
)
def get_note(
    note_id: str = Path(..., min_length=1, description="Note ID."),
) -> NoteOut:
    # PUBLIC_INTERFACE
    """Fetch a note by id."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Note not found")
        return _row_to_note_out(conn, row)


@app.put(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Update note",
    description="Update a note by ID, replacing title/content and optionally replacing tags.",
    operation_id="update_note",
)
def update_note(
    payload: NoteUpdate,
    note_id: str = Path(..., min_length=1, description="Note ID."),
) -> NoteOut:
    # PUBLIC_INTERFACE
    """Update an existing note and optionally replace its tags."""
    with _db() as conn:
        existing = conn.execute("SELECT id FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Note not found")

        now = _utc_now_iso()
        conn.execute(
            """
            UPDATE notes
            SET title = ?, content = ?, updated_at = ?
            WHERE id = ?
            """,
            (payload.title, payload.content, now, note_id),
        )

        _set_note_tags(conn, note_id, payload.tags)

        row = conn.execute(
            "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Failed to load updated note")
        return _row_to_note_out(conn, row)


@app.delete(
    "/notes/{note_id}",
    response_model=DeleteResult,
    tags=["Notes"],
    summary="Delete note",
    description="Delete a note by ID. Cascades to note_tags via FK.",
    operation_id="delete_note",
)
def delete_note(
    note_id: str = Path(..., min_length=1, description="Note ID."),
) -> DeleteResult:
    # PUBLIC_INTERFACE
    """Delete a note by id."""
    with _db() as conn:
        res = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Note not found")
        return DeleteResult(deleted=True, id=note_id)
