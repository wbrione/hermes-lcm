
from __future__ import annotations
import os, sqlite3
from pathlib import Path
from hermes_lcm.store import MessageStore

def _wal_exists(db_path):
    return os.path.exists(str(db_path) + "-wal")

def test_store_shutdown_wal_survives_for_raw_connections(tmp_path):
    """Episode-8 trigger (2026-10-02): auxiliary raw sqlite connections
    (trajectory/lifecycle/embedding-read) hold the database open while all
    MessageStore instances shut down; the per-instance keeper dies with the
    last store and SQLite unlinks -wal under the raw connections."""
    db = tmp_path / "episode8.db"
    raw = sqlite3.connect(str(db))   # simulates trajectory/embedding raw conn
    s1 = MessageStore(db)
    assert _wal_exists(db)
    s1.shutdown()                    # last store closes -> keeper dies
    assert _wal_exists(db), "WAL unlinked while raw connection still open (episode-8 trigger, issue #628)"
    raw.close()
