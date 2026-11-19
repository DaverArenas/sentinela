"""SQLite event store and snapshot evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinela_runtime.core.config import SnapshotConfig
from sentinela_runtime.events.schemas import (
    BoundingBox,
    Event,
    EventType,
    EvidenceInfo,
    ObjectInfo,
    ProcessingInfo,
    ZoneInfo,
)
from sentinela_runtime.storage import sqlite_store
from sentinela_runtime.storage.snapshots import SnapshotStore, free_disk_mib, write_durably
from sentinela_runtime.storage.sqlite_store import DB_SCHEMA_VERSION, SqliteEventStore, StorageError

T0 = datetime(2026, 11, 19, 10, 0, 0, tzinfo=UTC)
IMAGE = np.full((48, 64, 3), 120, dtype=np.uint8)


def zone_event(event_type: EventType = EventType.ZONE_ENTRY, second: int = 0, **overrides: object) -> Event:
    values: dict[str, object] = dict(
        event_type=event_type,
        timestamp=T0.replace(second=second),
        node_id="edge-node-001",
        camera_id="camera-001",
        object=ObjectInfo("person", 0, 0.9, BoundingBox(1, 2, 3, 4), track_id="track-7"),
        zone=ZoneInfo("zone-001", "Restricted Area"),
        processing=ProcessingInfo("m", "1", None, "onnxruntime-cuda", "cfg", 12),
    )
    return Event(**{**values, **overrides})  # type: ignore[arg-type]


# --- SQLite -------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteEventStore]:
    with SqliteEventStore(tmp_path / "data" / "events.db", clock=lambda: T0) as opened:
        yield opened


def test_store_persists_the_full_payload_and_indexed_columns(store: SqliteEventStore, tmp_path: Path) -> None:
    event = zone_event(evidence=EvidenceInfo("snapshots/2026-11-19/x.jpg"))

    store.publish(event)

    assert store.get(event.event_id) == event.to_dict()
    with sqlite3.connect(tmp_path / "data" / "events.db") as conn:
        row = conn.execute(
            "SELECT event_type, track_id, zone_id, snapshot_path, stored_at FROM events"
        ).fetchone()
    assert row == (
        "zone_entry",
        "track-7",
        "zone-001",
        "snapshots/2026-11-19/x.jpg",
        "2026-11-19T10:00:00.000Z",
    )


def test_publish_is_idempotent_per_event_id(store: SqliteEventStore) -> None:
    event = zone_event()

    store.publish(event)
    store.publish(event)  # retry of the same event

    assert store.count() == 1


def test_recent_and_count_queries(store: SqliteEventStore) -> None:
    store.publish(zone_event(second=1))
    store.publish(zone_event(EventType.ZONE_EXIT, second=5, zone=None))

    recent = store.recent(limit=1)

    assert [e["event_type"] for e in recent] == ["zone_exit"]
    assert recent[0]["zone"] is None
    assert (store.count(), store.count("zone_entry"), store.count("zone_exit")) == (2, 1, 1)
    assert store.get("evt-missing") is None


def test_events_survive_reopening_and_schema_is_versioned(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    event = zone_event()
    with SqliteEventStore(path) as first:
        first.publish(event)

    with SqliteEventStore(path) as second:
        assert second.get(event.event_id) is not None
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_database_from_a_newer_runtime_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {DB_SCHEMA_VERSION + 1}")

    with pytest.raises(StorageError, match="newer than supported"):
        SqliteEventStore(path)


def test_unopenable_database_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")

    with pytest.raises(StorageError, match="Cannot open event store"):
        SqliteEventStore(blocker / "events.db")


BROKEN_SQL = "INSERT INTO missing_table VALUES (?)"
NULL_SCHEMA_SQL = sqlite_store.INSERT_SQL.replace(
    "VALUES (?, ?,", "VALUES (?, NULLIF(?, '1.0'),"
)  # NOT NULL violation: a permanent error


def test_failed_write_is_buffered_and_retried_on_next_publish(
    store: SqliteEventStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = zone_event(second=1), zone_event(second=2)
    monkeypatch.setattr(sqlite_store, "INSERT_SQL", BROKEN_SQL)
    with caplog.at_level("WARNING", logger="sentinela"):
        store.publish(first)  # disk full / locked: must not raise
    assert (store.pending_count, store.count()) == (1, 0)
    assert "buffered for retry" in caplog.text

    monkeypatch.undo()
    store.publish(second)

    assert (store.pending_count, store.count()) == (0, 2)
    assert [e["event_id"] for e in store.recent()] == [second.event_id, first.event_id]


def test_retry_buffer_is_bounded(
    store: SqliteEventStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sqlite_store, "INSERT_SQL", BROKEN_SQL)
    monkeypatch.setattr(sqlite_store, "MAX_PENDING_EVENTS", 2)

    with caplog.at_level("ERROR", logger="sentinela"):
        for second in range(3):
            store.publish(zone_event(second=second))

    assert store.pending_count == 2
    assert "retry buffer full" in caplog.text


def test_permanently_invalid_row_is_rejected_without_blocking_others(
    store: SqliteEventStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sqlite_store, "INSERT_SQL", NULL_SCHEMA_SQL)
    with caplog.at_level("ERROR", logger="sentinela"):
        store.publish(zone_event())

    assert (store.pending_count, store.count()) == (0, 0)
    assert "rejected by the event store" in caplog.text


def test_close_reports_events_that_could_not_be_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sqlite_store, "INSERT_SQL", BROKEN_SQL)
    event = zone_event()

    with (
        caplog.at_level("ERROR", logger="sentinela"),
        SqliteEventStore(tmp_path / "events.db") as opened,
    ):
        opened.publish(event)

    assert "could not be stored before shutdown" in caplog.text


def test_corrupt_database_file_raises_storage_error(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    path.write_bytes(b"this is not a sqlite database" * 100)

    with pytest.raises(StorageError, match="Cannot open event store"):
        SqliteEventStore(path)


# --- snapshots ------------------------------------------------------------------------------------

SNAPSHOTS = SnapshotConfig(enabled=True, event_types=("zone_entry",), jpeg_quality=90, min_free_disk_mb=10)


def test_snapshot_is_written_for_configured_event_types(tmp_path: Path) -> None:
    snapshots = SnapshotStore(tmp_path, SNAPSHOTS, free_space=lambda _p: 1e6)
    event = zone_event()

    relative = snapshots.capture(event, IMAGE)

    assert relative == f"snapshots/2026-11-19/{event.event_id}.jpg"
    saved = cv2.imread(str(tmp_path / relative))
    assert saved.shape == IMAGE.shape
    assert not list((tmp_path / "snapshots" / "2026-11-19").glob(".*.tmp"))  # temp file renamed away


@pytest.mark.parametrize(
    ("config", "event_type"),
    [
        (SNAPSHOTS, EventType.ZONE_EXIT),
        (replace(SNAPSHOTS, enabled=False), EventType.ZONE_ENTRY),
    ],
)
def test_snapshot_skipped_when_not_configured(
    tmp_path: Path, config: SnapshotConfig, event_type: EventType
) -> None:
    snapshots = SnapshotStore(tmp_path, config, free_space=lambda _p: 1e6)

    assert snapshots.capture(zone_event(event_type), IMAGE) is None
    assert not (tmp_path / "snapshots").exists()


def test_snapshot_skipped_when_disk_is_low(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    snapshots = SnapshotStore(tmp_path, SNAPSHOTS, free_space=lambda _p: 5.0)

    with caplog.at_level("WARNING", logger="sentinela"):
        assert snapshots.capture(zone_event(), IMAGE) is None

    assert "Low disk space" in caplog.text


def test_snapshot_encoding_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentinela_runtime.storage.snapshots.cv2.imencode", lambda *_a: (False, None))
    snapshots = SnapshotStore(tmp_path, SNAPSHOTS, free_space=lambda _p: 1e6)

    with pytest.raises(OSError, match="Could not encode snapshot"):
        snapshots.capture(zone_event(), IMAGE)


def test_durable_write_removes_the_temp_file_when_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(*_a: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr("sentinela_runtime.storage.snapshots.os.replace", failing_replace)

    with pytest.raises(OSError, match="rename failed"):
        write_durably(tmp_path / "x.jpg", b"jpeg")

    assert list(tmp_path.iterdir()) == []


def test_free_disk_mib_reports_a_positive_value(tmp_path: Path) -> None:
    assert free_disk_mib(tmp_path) > 0
