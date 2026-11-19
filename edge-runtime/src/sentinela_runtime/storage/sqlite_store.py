"""SQLite event repository: the local, durable record of every event (``EventSink``).

* One row per event, keyed by ``event_id``. Inserts are idempotent (a duplicate ``event_id`` is
  ignored), so a retried publish never duplicates an event and IDs survive future upstream sync.
* A failed write (disk full, database locked) does not lose the event: it stays in a bounded
  in-memory buffer and is retried on the next publish and on close.
* The full schema-1.0 JSON is kept verbatim in ``payload``; the other columns are indexed
  copies for querying.
* The database schema is versioned with ``PRAGMA user_version`` and migrated forward on open.
* WAL journaling with ``synchronous=FULL``: a committed event survives process crashes and
  power loss (event volume is low, so the extra fsync is affordable).

Inspect from ``edge-runtime/`` with Python's built-in SQLite shell (Python 3.12+)::

    uv run python -m sqlite3 ../data/sentinela.db \
        "SELECT timestamp, event_type, track_id, zone_id, snapshot_path FROM events ORDER BY timestamp"
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from sentinela_runtime.core.log import get_logger
from sentinela_runtime.events.schemas import Event, format_timestamp

log = get_logger("storage")

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE events (
        event_id       TEXT PRIMARY KEY,
        schema_version TEXT NOT NULL,
        event_type     TEXT NOT NULL,
        timestamp      TEXT NOT NULL,
        node_id        TEXT NOT NULL,
        camera_id      TEXT NOT NULL,
        object_class   TEXT,
        track_id       TEXT,
        confidence     REAL,
        zone_id        TEXT,
        snapshot_path  TEXT,
        payload        TEXT NOT NULL,
        stored_at      TEXT NOT NULL
    );
    CREATE INDEX idx_events_timestamp ON events (timestamp);
    CREATE INDEX idx_events_type_zone ON events (event_type, zone_id);
    """,
)
DB_SCHEMA_VERSION = len(MIGRATIONS)

INSERT_SQL = """
    INSERT INTO events (
        event_id, schema_version, event_type, timestamp, node_id, camera_id, object_class,
        track_id, confidence, zone_id, snapshot_path, payload, stored_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (event_id) DO NOTHING
"""
MAX_PENDING_EVENTS = 1_000  # bounded retry buffer for writes that failed transiently
Row = tuple[Any, ...]


class StorageError(RuntimeError):
    """Raised when the local event store cannot be opened or written."""


class SqliteEventStore:
    def __init__(self, path: Path, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._path = path
        self._clock = clock
        self._pending: deque[Row] = deque()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Cannot open event store {path}: {exc}") from exc
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._migrate()
        except (StorageError, sqlite3.Error) as exc:
            self._conn.close()
            if isinstance(exc, StorageError):
                raise
            raise StorageError(f"Cannot open event store {path}: {exc}") from exc
        log.info("Event store opened", extra={"path": str(path), "db_schema_version": DB_SCHEMA_VERSION})

    def _migrate(self) -> None:
        current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if current > DB_SCHEMA_VERSION:
            raise StorageError(
                f"Event store {self._path} has schema v{current}, newer than supported v{DB_SCHEMA_VERSION}"
            )
        for version in range(current, DB_SCHEMA_VERSION):
            self._conn.executescript(
                f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._flush()
        if self._pending:
            log.error(
                "Events could not be stored before shutdown",
                extra={"lost_event_ids": [row[0] for row in self._pending]},
            )
        self._conn.close()

    @property
    def pending_count(self) -> int:
        """Events waiting for a successful write."""
        return len(self._pending)

    def publish(self, event: Event) -> None:
        """Persist one event; on a transient failure buffer it and retry on the next publish."""
        if len(self._pending) >= MAX_PENDING_EVENTS:
            dropped = self._pending.popleft()
            log.error(
                "Event retry buffer full; dropping oldest unsaved event", extra={"event_id": dropped[0]}
            )
        self._pending.append(self._row(event))
        self._flush()

    def _flush(self) -> None:
        """Write buffered rows in order, one transaction each; stop at the first transient failure."""
        while self._pending:
            row = self._pending[0]
            try:
                with self._conn:  # commits, or rolls back on error
                    self._conn.execute(INSERT_SQL, row)
            except sqlite3.IntegrityError as exc:  # permanent: retrying cannot succeed
                log.error("Event rejected by the event store", extra={"event_id": row[0], "error": str(exc)})
            except sqlite3.Error as exc:
                log.warning(
                    "Event store write failed; events buffered for retry",
                    extra={"pending": len(self._pending), "error": str(exc)},
                )
                return
            self._pending.popleft()

    def _row(self, event: Event) -> Row:
        obj, zone, evidence = event.object, event.zone, event.evidence
        return (
            event.event_id,
            event.schema_version,
            str(event.event_type),
            format_timestamp(event.timestamp),
            event.node_id,
            event.camera_id,
            obj.class_name,
            obj.track_id,
            obj.confidence,
            zone.zone_id if zone else None,
            evidence.snapshot_path if evidence else None,
            event.to_json(),
            format_timestamp(self._clock()),
        )

    def get(self, event_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT payload FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent events first (by event timestamp)."""
        rows = self._conn.execute(
            "SELECT payload FROM events ORDER BY timestamp DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(payload) for (payload,) in rows]

    def count(self, event_type: str | None = None) -> int:
        if event_type is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        query = "SELECT COUNT(*) FROM events WHERE event_type = ?"
        return int(self._conn.execute(query, (event_type,)).fetchone()[0])
