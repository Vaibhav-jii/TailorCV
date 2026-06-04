"""
Store Intelligence Database — SQLite storage layer for events, journeys, and metrics.

Uses SQLite for zero-setup, file-based persistence. In production,
this would migrate to TimescaleDB (time-series) + PostgreSQL (relational).

ADR: SQLite chosen over PostgreSQL/TimescaleDB for hackathon scope.
- Zero setup, single file, great for demos
- Sufficient for the data scale (5 videos)
- Same SQL interface — migration path is clear
"""

import sqlite3
import json
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import StoreEvent, PersonJourney, AnomalyAlert
from config.settings import settings

logger = logging.getLogger("store_intelligence.database")


class Database:
    """SQLite storage for store intelligence data."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        """Create tables and indexes."""
        # Migrate existing schema if columns are missing
        if Path(self.db_path).exists():
            try:
                with self._get_conn() as conn:
                    # Check events columns
                    cursor = conn.execute("PRAGMA table_info(events)")
                    events_cols = [row["name"] for row in cursor.fetchall()]
                    if events_cols:
                        if "is_staff" not in events_cols:
                            conn.execute("ALTER TABLE events ADD COLUMN is_staff INTEGER DEFAULT 0")
                            logger.info("Migrated events: added is_staff")
                        if "confidence" not in events_cols:
                            conn.execute("ALTER TABLE events ADD COLUMN confidence REAL DEFAULT 0.0")
                        if "session_seq" not in events_cols:
                            conn.execute("ALTER TABLE events ADD COLUMN session_seq INTEGER")
                    
                    # Check journeys columns
                    cursor = conn.execute("PRAGMA table_info(journeys)")
                    journeys_cols = [row["name"] for row in cursor.fetchall()]
                    if journeys_cols:
                        if "is_staff" not in journeys_cols:
                            conn.execute("ALTER TABLE journeys ADD COLUMN is_staff INTEGER DEFAULT 0")
                            logger.info("Migrated journeys: added is_staff")
            except sqlite3.OperationalError as e:
                logger.warning(f"Migration error (ignoring): {e}")

        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    camera_id TEXT DEFAULT 'CAM_01',
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER DEFAULT 0,
                    
                    -- Entry/Exit fields
                    id_token TEXT,
                    person_id TEXT,
                    is_staff INTEGER DEFAULT 0,
                    gender TEXT,
                    age INTEGER,
                    age_bucket TEXT,
                    is_face_hidden INTEGER DEFAULT 0,
                    group_id TEXT,
                    group_size INTEGER,
                    
                    -- Zone fields
                    track_id INTEGER,
                    zone_id TEXT,
                    zone_name TEXT,
                    zone_type TEXT,
                    is_revenue_zone TEXT,
                    zone_hotspot_x REAL,
                    zone_hotspot_y REAL,
                    dwell_ms INTEGER,
                    
                    -- Queue fields
                    queue_event_id TEXT,
                    queue_join_ts TEXT,
                    queue_served_ts TEXT,
                    queue_exit_ts TEXT,
                    wait_seconds INTEGER,
                    queue_position_at_join INTEGER,
                    abandoned INTEGER,
                    
                    -- Metadata
                    confidence REAL DEFAULT 0.0,
                    session_seq INTEGER,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS journeys (
                    person_id TEXT PRIMARY KEY,
                    track_id INTEGER,
                    store_id TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    zones_visited_json TEXT DEFAULT '[]',
                    total_dwell_time REAL DEFAULT 0.0,
                    reached_billing INTEGER DEFAULT 0,
                    purchased INTEGER DEFAULT 0,
                    abandoned_billing INTEGER DEFAULT 0,
                    is_reentry INTEGER DEFAULT 0,
                    is_staff INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 0,
                    gender TEXT,
                    age INTEGER,
                    age_bucket TEXT,
                    group_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS anomalies (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT,
                    current_value REAL,
                    expected_range TEXT,
                    message TEXT,
                    suggested_action TEXT DEFAULT '',
                    store_id TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS pos_transactions (
                    order_id INTEGER,
                    order_date TEXT,
                    order_time TEXT,
                    store_id TEXT,
                    product_id TEXT,
                    brand_name TEXT,
                    total_amount REAL,
                    timestamp TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS processing_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT,
                    camera_id TEXT,
                    store_id TEXT,
                    total_frames INTEGER,
                    processed_frames INTEGER,
                    total_events INTEGER,
                    metrics_json TEXT,
                    started_at TEXT,
                    completed_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Indexes for fast querying
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id);
                CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
                CREATE INDEX IF NOT EXISTS idx_events_store ON events(store_id);
                CREATE INDEX IF NOT EXISTS idx_events_is_staff ON events(is_staff);
                CREATE INDEX IF NOT EXISTS idx_journeys_purchased ON journeys(purchased);
                CREATE INDEX IF NOT EXISTS idx_journeys_store ON journeys(store_id);
                CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(anomaly_type);
                CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
                CREATE INDEX IF NOT EXISTS idx_anomalies_store ON anomalies(store_id);
                CREATE INDEX IF NOT EXISTS idx_pos_store ON pos_transactions(store_id);
                CREATE INDEX IF NOT EXISTS idx_pos_timestamp ON pos_transactions(timestamp);
            """)

    # ═══ Event Operations ═══

    def insert_event(self, event: StoreEvent):
        """Insert a single event."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, event_type, store_id, camera_id, timestamp, frame_number,
                    id_token, person_id, is_staff, gender, age, age_bucket,
                    is_face_hidden, group_id, group_size,
                    track_id, zone_id, zone_name, zone_type, is_revenue_zone,
                    zone_hotspot_x, zone_hotspot_y, dwell_ms,
                    queue_event_id, queue_join_ts, queue_served_ts, queue_exit_ts,
                    wait_seconds, queue_position_at_join, abandoned,
                    confidence, session_seq, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._event_to_row(event),
            )

    def insert_events_batch(self, events: List[StoreEvent]) -> dict:
        """
        Batch insert events — idempotent by event_id.
        Returns: {"accepted": N, "duplicates": N, "errors": []}
        """
        if not events:
            return {"accepted": 0, "duplicates": 0, "errors": []}
        
        accepted = 0
        duplicates = 0
        errors = []
        
        with self._get_conn() as conn:
            for event in events:
                try:
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO events
                           (event_id, event_type, store_id, camera_id, timestamp, frame_number,
                            id_token, person_id, is_staff, gender, age, age_bucket,
                            is_face_hidden, group_id, group_size,
                            track_id, zone_id, zone_name, zone_type, is_revenue_zone,
                            zone_hotspot_x, zone_hotspot_y, dwell_ms,
                            queue_event_id, queue_join_ts, queue_served_ts, queue_exit_ts,
                            wait_seconds, queue_position_at_join, abandoned,
                            confidence, session_seq, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        self._event_to_row(event),
                    )
                    if cursor.rowcount > 0:
                        accepted += 1
                    else:
                        duplicates += 1
                except Exception as e:
                    errors.append({"event_id": event.event_id, "error": str(e)})
        
        return {"accepted": accepted, "duplicates": duplicates, "errors": errors}

    def ingest_raw_events(self, raw_events: List[dict]) -> dict:
        """
        Ingest raw event dicts (from POST /events/ingest).
        Validates, deduplicates, stores. Returns partial success info.
        """
        accepted = 0
        duplicates = 0
        errors = []
        
        with self._get_conn() as conn:
            for i, raw in enumerate(raw_events):
                try:
                    event_id = raw.get("event_id") or raw.get("queue_event_id") or str(__import__("uuid").uuid4())
                    event_type = raw.get("event_type", "")
                    store_id = raw.get("store_id") or raw.get("store_code", "UNKNOWN")
                    camera_id = raw.get("camera_id", "")
                    timestamp = raw.get("event_timestamp") or raw.get("event_time") or raw.get("queue_join_ts", "")
                    
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO events
                           (event_id, event_type, store_id, camera_id, timestamp,
                            id_token, person_id, is_staff, gender, age, age_bucket,
                            is_face_hidden, group_id, group_size,
                            track_id, zone_id, zone_name, zone_type, is_revenue_zone,
                            zone_hotspot_x, zone_hotspot_y, dwell_ms,
                            queue_event_id, queue_join_ts, queue_served_ts, queue_exit_ts,
                            wait_seconds, queue_position_at_join, abandoned,
                            metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            event_id, event_type, store_id, camera_id, timestamp,
                            raw.get("id_token"), raw.get("person_id") or raw.get("id_token"),
                            1 if raw.get("is_staff") else 0,
                            raw.get("gender_pred") or raw.get("gender"),
                            raw.get("age_pred") or raw.get("age"),
                            raw.get("age_bucket"),
                            1 if raw.get("is_face_hidden") else 0,
                            raw.get("group_id"),
                            raw.get("group_size"),
                            raw.get("track_id"),
                            raw.get("zone_id"),
                            raw.get("zone_name"),
                            raw.get("zone_type"),
                            raw.get("is_revenue_zone"),
                            raw.get("zone_hotspot_x"),
                            raw.get("zone_hotspot_y"),
                            raw.get("dwell_ms"),
                            raw.get("queue_event_id"),
                            raw.get("queue_join_ts"),
                            raw.get("queue_served_ts"),
                            raw.get("queue_exit_ts"),
                            raw.get("wait_seconds"),
                            raw.get("queue_position_at_join"),
                            1 if raw.get("abandoned") else (0 if raw.get("abandoned") is not None else None),
                            json.dumps({k: v for k, v in raw.items() if k not in (
                                "event_id", "event_type", "store_id", "camera_id", "timestamp"
                            )}, default=str),
                        ),
                    )
                    if cursor.rowcount > 0:
                        accepted += 1
                    else:
                        duplicates += 1
                except Exception as e:
                    errors.append({"index": i, "event_id": raw.get("event_id", ""), "error": str(e)})
        
        return {"accepted": accepted, "duplicates": duplicates, "errors": errors}

    def _event_to_row(self, event: StoreEvent) -> tuple:
        # Prefer the datetime timestamp, fall back to string fields
        if event.timestamp:
            ts = event.timestamp.isoformat()
        else:
            ts = event.event_timestamp or event.event_time or datetime.now().isoformat()
        return (
            event.event_id,
            event.event_type.value,
            event.store_id,
            event.camera_id,
            ts,
            event.frame_number,
            event.id_token or event.person_id,
            event.person_id,
            1 if event.is_staff else 0,
            event.gender_pred or event.gender,
            event.age_pred or event.age,
            event.age_bucket,
            1 if event.is_face_hidden else 0,
            event.group_id,
            event.group_size,
            event.track_id,
            event.zone_id,
            event.zone_name,
            event.zone_type,
            event.is_revenue_zone,
            event.zone_hotspot_x,
            event.zone_hotspot_y,
            event.dwell_ms,
            event.queue_event_id,
            event.queue_join_ts,
            event.queue_served_ts,
            event.queue_exit_ts,
            event.wait_seconds,
            event.queue_position_at_join,
            1 if event.abandoned else (0 if event.abandoned is not None else None),
            event.confidence,
            event.session_seq,
            json.dumps(event.metadata, default=str),
        )

    def get_events(
        self,
        event_type: Optional[str] = None,
        person_id: Optional[str] = None,
        store_id: Optional[str] = None,
        zone: Optional[str] = None,
        camera_id: Optional[str] = None,
        since: Optional[str] = None,
        exclude_staff: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Query events with optional filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if person_id:
            query += " AND (person_id = ? OR id_token = ?)"
            params.extend([person_id, person_id])
        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)
        if zone:
            query += " AND (zone_name = ? OR zone_id = ?)"
            params.extend([zone, zone])
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if exclude_staff:
            query += " AND is_staff = 0"

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_event_counts(self, store_id: Optional[str] = None) -> Dict[str, int]:
        """Get count of events by type."""
        with self._get_conn() as conn:
            query = "SELECT event_type, COUNT(*) as count FROM events"
            params = []
            if store_id:
                query += " WHERE store_id = ?"
                params.append(store_id)
            query += " GROUP BY event_type"
            rows = conn.execute(query, params).fetchall()
            return {row["event_type"]: row["count"] for row in rows}

    # ═══ Journey Operations ═══

    def insert_journey(self, journey: PersonJourney):
        """Insert or update a person journey."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO journeys
                   (person_id, track_id, store_id, entry_time, exit_time, zones_visited_json,
                    total_dwell_time, reached_billing, purchased, abandoned_billing,
                    is_reentry, is_staff, event_count, gender, age, age_bucket, group_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    journey.person_id,
                    journey.track_id,
                    getattr(journey, 'store_id', 'STORE_001'),
                    journey.entry_time.isoformat() if journey.entry_time else None,
                    journey.exit_time.isoformat() if journey.exit_time else None,
                    json.dumps(journey.zones_visited, default=str),
                    journey.total_dwell_time,
                    int(journey.reached_billing),
                    int(journey.purchased),
                    int(journey.abandoned_billing),
                    int(journey.is_reentry),
                    int(journey.is_staff),
                    journey.event_count,
                    journey.gender,
                    journey.age,
                    journey.age_bucket,
                    journey.group_id,
                ),
            )

    def insert_journeys_batch(self, journeys: List[PersonJourney]):
        """Batch insert journeys."""
        for j in journeys:
            self.insert_journey(j)

    def get_journeys(self, purchased_only: bool = False, store_id: Optional[str] = None,
                     exclude_staff: bool = False) -> List[dict]:
        """Get all journeys, optionally filtered."""
        query = "SELECT * FROM journeys WHERE 1=1"
        params = []
        if purchased_only:
            query += " AND purchased = 1"
        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)
        if exclude_staff:
            query += " AND is_staff = 0"
        query += " ORDER BY entry_time DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get("zones_visited_json"):
                    d["zones_visited"] = json.loads(d["zones_visited_json"])
                else:
                    d["zones_visited"] = []
                result.append(d)
            return result

    # ═══ Store Metrics (GET /stores/{id}/metrics) ═══
    
    def get_store_metrics(self, store_id: str) -> dict:
        """
        Get real-time metrics for a store.
        Excludes is_staff=true from customer metrics.
        """
        with self._get_conn() as conn:
            # Unique visitors (non-staff entries)
            unique_visitors = conn.execute(
                """SELECT COUNT(DISTINCT COALESCE(person_id, id_token)) 
                   FROM events WHERE store_id = ? AND event_type = 'entry' AND is_staff = 0""",
                (store_id,)
            ).fetchone()[0] or 0
            
            # Conversion rate via POS correlation
            pos_count = conn.execute(
                "SELECT COUNT(DISTINCT order_id) FROM pos_transactions WHERE store_id = ?",
                (store_id,)
            ).fetchone()[0] or 0
            
            conversion_rate = (pos_count / unique_visitors) if unique_visitors > 0 else 0.0
            
            # Avg dwell per zone
            zone_dwell = conn.execute(
                """SELECT zone_name, 
                   AVG(dwell_ms) as avg_dwell_ms,
                   COUNT(*) as visit_count
                   FROM events 
                   WHERE store_id = ? AND event_type IN ('zone_exited', 'zone_dwell') 
                   AND is_staff = 0 AND zone_name IS NOT NULL
                   GROUP BY zone_name""",
                (store_id,)
            ).fetchall()
            
            avg_dwell_per_zone = {}
            for row in zone_dwell:
                avg_dwell_per_zone[row["zone_name"]] = {
                    "avg_dwell_ms": row["avg_dwell_ms"] or 0,
                    "visit_count": row["visit_count"],
                }
            
            # Queue depth (latest)
            queue_data = conn.execute(
                """SELECT queue_position_at_join FROM events 
                   WHERE store_id = ? AND event_type IN ('queue_joined', 'queue_completed', 'queue_abandoned')
                   ORDER BY timestamp DESC LIMIT 1""",
                (store_id,)
            ).fetchone()
            queue_depth = queue_data["queue_position_at_join"] if queue_data and queue_data["queue_position_at_join"] else 0
            
            # Abandonment rate
            queue_total = conn.execute(
                "SELECT COUNT(*) FROM events WHERE store_id = ? AND event_type IN ('queue_completed', 'queue_abandoned')",
                (store_id,)
            ).fetchone()[0] or 0
            queue_abandoned = conn.execute(
                "SELECT COUNT(*) FROM events WHERE store_id = ? AND event_type = 'queue_abandoned'",
                (store_id,)
            ).fetchone()[0] or 0
            abandonment_rate = (queue_abandoned / queue_total) if queue_total > 0 else 0.0
            
            return {
                "store_id": store_id,
                "unique_visitors": unique_visitors,
                "conversion_rate": round(conversion_rate, 4),
                "avg_dwell_per_zone": avg_dwell_per_zone,
                "queue_depth": queue_depth,
                "abandonment_rate": round(abandonment_rate, 4),
                "pos_transactions": pos_count,
                "generated_at": datetime.now().isoformat(),
            }

    # ═══ Funnel (GET /stores/{id}/funnel) ═══
    
    def get_conversion_funnel(self, store_id: Optional[str] = None) -> dict:
        """Get conversion funnel metrics — the North Star."""
        with self._get_conn() as conn:
            where = "WHERE is_staff = 0"
            params = []
            if store_id:
                where += " AND store_id = ?"
                params.append(store_id)
            
            total = conn.execute(f"SELECT COUNT(*) FROM journeys {where}", params).fetchone()[0]
            browsed = conn.execute(
                f"SELECT COUNT(*) FROM journeys {where} AND zones_visited_json != '[]'", params
            ).fetchone()[0]
            reached_billing = conn.execute(
                f"SELECT COUNT(*) FROM journeys {where} AND reached_billing = 1", params
            ).fetchone()[0]
            purchased = conn.execute(
                f"SELECT COUNT(*) FROM journeys {where} AND purchased = 1", params
            ).fetchone()[0]
            abandoned = conn.execute(
                f"SELECT COUNT(*) FROM journeys {where} AND abandoned_billing = 1", params
            ).fetchone()[0]

            # Drop-off percentages
            entry_to_browse = (browsed / total * 100) if total > 0 else 0
            browse_to_billing = (reached_billing / browsed * 100) if browsed > 0 else 0
            billing_to_purchase = (purchased / reached_billing * 100) if reached_billing > 0 else 0

            return {
                "store_id": store_id,
                "funnel": [
                    {"stage": "Entry", "count": total, "dropoff_pct": 0},
                    {"stage": "Zone Visit", "count": browsed, "dropoff_pct": round(100 - entry_to_browse, 1)},
                    {"stage": "Billing Queue", "count": reached_billing, "dropoff_pct": round(100 - browse_to_billing, 1)},
                    {"stage": "Purchase", "count": purchased, "dropoff_pct": round(100 - billing_to_purchase, 1)},
                ],
                "conversion_rate": round(purchased / total, 4) if total > 0 else 0,
                "billing_reach_rate": round(reached_billing / total, 4) if total > 0 else 0,
                "browse_rate": round(browsed / total, 4) if total > 0 else 0,
                "abandon_rate": round(abandoned / reached_billing, 4) if reached_billing > 0 else 0,
            }

    # ═══ Heatmap (GET /stores/{id}/heatmap) ═══
    
    def get_heatmap(self, store_id: str) -> dict:
        """Zone visit frequency + avg dwell, normalised 0-100."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT 
                    zone_name, zone_id, zone_type,
                    COUNT(*) as visit_count,
                    COUNT(DISTINCT COALESCE(person_id, id_token)) as unique_visitors,
                    AVG(dwell_ms) as avg_dwell_ms
                FROM events 
                WHERE store_id = ? AND event_type IN ('zone_entered', 'zone_exited', 'zone_dwell')
                AND is_staff = 0 AND zone_name IS NOT NULL
                GROUP BY zone_name
            """, (store_id,)).fetchall()
            
            if not rows:
                return {"store_id": store_id, "zones": [], "data_confidence": "LOW"}
            
            zones = [dict(r) for r in rows]
            max_visits = max(z["visit_count"] for z in zones) if zones else 1
            
            total_sessions = conn.execute(
                "SELECT COUNT(*) FROM journeys WHERE store_id = ? AND is_staff = 0", (store_id,)
            ).fetchone()[0] or 0
            
            for z in zones:
                z["normalised_score"] = round((z["visit_count"] / max_visits) * 100, 1) if max_visits > 0 else 0
                z["avg_dwell_ms"] = round(z["avg_dwell_ms"] or 0, 0)
            
            return {
                "store_id": store_id,
                "zones": zones,
                "data_confidence": "HIGH" if total_sessions >= 20 else "LOW",
                "total_sessions": total_sessions,
            }

    # ═══ Zone Analytics ═══
    
    def get_zone_analytics(self, store_id: Optional[str] = None) -> List[dict]:
        """Get per-zone visit analytics."""
        with self._get_conn() as conn:
            where = "WHERE event_type IN ('zone_exited', 'zone_dwell', 'queue_completed', 'queue_abandoned') AND zone_name IS NOT NULL AND is_staff = 0"
            params = []
            if store_id:
                where += " AND store_id = ?"
                params.append(store_id)
            
            rows = conn.execute(f"""
                SELECT 
                    zone_name,
                    COUNT(*) as visit_count,
                    COUNT(DISTINCT COALESCE(person_id, id_token)) as unique_visitors,
                    AVG(dwell_ms) as avg_dwell_ms,
                    MAX(dwell_ms) as max_dwell_ms,
                    MIN(dwell_ms) as min_dwell_ms
                FROM events 
                {where}
                GROUP BY zone_name
                ORDER BY visit_count DESC
            """, params).fetchall()
            
            results = []
            for row in rows:
                d = dict(row)
                d["zone"] = row["zone_name"]
                # Convert milliseconds to seconds for dashboard
                d["avg_dwell"] = round((row["avg_dwell_ms"] or 0) / 1000.0, 2)
                d["max_dwell"] = round((row["max_dwell_ms"] or 0) / 1000.0, 2)
                d["min_dwell"] = round((row["min_dwell_ms"] or 0) / 1000.0, 2)
                results.append(d)
            return results

    def get_hourly_traffic(self, store_id: Optional[str] = None) -> List[dict]:
        """Get entry count aggregated by hour."""
        with self._get_conn() as conn:
            where = "WHERE event_type = 'entry' AND is_staff = 0"
            params = []
            if store_id:
                where += " AND store_id = ?"
                params.append(store_id)
            rows = conn.execute(f"""
                SELECT 
                    substr(timestamp, 12, 2) as hour,
                    COUNT(*) as entries
                FROM events 
                {where}
                GROUP BY substr(timestamp, 12, 2)
                ORDER BY hour
            """, params).fetchall()
            return [dict(row) for row in rows]

    def get_queue_metrics(self, store_id: Optional[str] = None) -> dict:
        """Get billing queue analytics."""
        with self._get_conn() as conn:
            where = "WHERE is_staff = 0"
            params = []
            if store_id:
                where += " AND store_id = ?"
                params.append(store_id)
            
            queue_joins = conn.execute(
                f"SELECT COUNT(*) FROM events {where} AND event_type IN ('queue_joined', 'queue_completed', 'queue_abandoned')",
                params
            ).fetchone()[0]
            queue_abandons = conn.execute(
                f"SELECT COUNT(*) FROM events {where} AND event_type = 'queue_abandoned'",
                params
            ).fetchone()[0]
            purchases = conn.execute(
                f"SELECT COUNT(*) FROM events {where} AND event_type = 'queue_completed'",
                params
            ).fetchone()[0]

            avg_wait = conn.execute(f"""
                SELECT AVG(wait_seconds)
                FROM events 
                {where} AND event_type IN ('queue_completed', 'queue_abandoned')
            """, params).fetchone()[0]

            return {
                "total_queue_joins": queue_joins,
                "total_queue_abandons": queue_abandons,
                "total_purchases": purchases,
                "abandon_rate": round(queue_abandons / queue_joins, 4) if queue_joins > 0 else 0,
                "avg_wait_seconds": round(avg_wait or 0, 1),
            }

    # ═══ Anomaly Operations ═══

    def insert_anomaly(self, alert: AnomalyAlert):
        """Insert an anomaly alert."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO anomalies
                   (alert_id, timestamp, anomaly_type, severity, metric_name,
                    current_value, expected_range, message, suggested_action, store_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.alert_id, alert.timestamp.isoformat(),
                    alert.anomaly_type, alert.severity, alert.metric_name,
                    alert.current_value, alert.expected_range, alert.message,
                    alert.suggested_action, alert.store_id,
                ),
            )

    def get_anomalies(self, store_id: Optional[str] = None, severity: Optional[str] = None, 
                      limit: int = 50) -> List[dict]:
        """Get recent anomaly alerts."""
        query = "SELECT * FROM anomalies WHERE 1=1"
        params: list = []
        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ═══ POS Operations ═══
    
    def load_pos_csv(self, csv_path: str, store_id: str = None):
        """Load POS transactions from CSV file."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                sid = store_id or row.get("store_id", "")
                order_date = row.get("order_date", "")
                order_time = row.get("order_time", "")
                # Convert DD-MM-YYYY to YYYY-MM-DD
                try:
                    parts = order_date.split("-")
                    if len(parts) == 3 and len(parts[2]) == 4:
                        iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    else:
                        iso_date = order_date
                except:
                    iso_date = order_date
                timestamp = f"{iso_date}T{order_time}" if order_time else iso_date
                
                rows.append((
                    int(row.get("order_id", 0)),
                    order_date,
                    order_time,
                    sid,
                    row.get("product_id", ""),
                    row.get("brand_name", ""),
                    float(row.get("total_amount", 0)),
                    timestamp,
                ))
        
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO pos_transactions
                   (order_id, order_date, order_time, store_id, product_id, brand_name, total_amount, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.info(f"Loaded {len(rows)} POS transactions from {csv_path}")
        return len(rows)

    # ═══ Health Check Data ═══
    
    def get_health_data(self) -> dict:
        """Get health check data including last event per store and stale feed detection."""
        with self._get_conn() as conn:
            # Last event per store
            rows = conn.execute("""
                SELECT store_id, MAX(timestamp) as last_event_ts, COUNT(*) as event_count
                FROM events GROUP BY store_id
            """).fetchall()
            
            stores = {}
            now = datetime.now()
            for row in rows:
                last_ts = row["last_event_ts"]
                is_stale = False
                try:
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00") if last_ts else "")
                    lag_minutes = (now - last_dt.replace(tzinfo=None)).total_seconds() / 60
                    is_stale = lag_minutes > 10
                except:
                    lag_minutes = None
                    is_stale = True
                
                stores[row["store_id"]] = {
                    "last_event_timestamp": last_ts,
                    "event_count": row["event_count"],
                    "status": "STALE_FEED" if is_stale else "ACTIVE",
                    "lag_minutes": round(lag_minutes, 1) if lag_minutes is not None else None,
                }
            
            return {
                "stores": stores,
                "total_events": sum(s["event_count"] for s in stores.values()),
            }

    # ═══ Processing Run Operations ═══

    def record_processing_run(self, results: dict):
        """Record a video processing run."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO processing_runs
                   (video_path, camera_id, store_id, total_frames,
                    processed_frames, total_events, metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    results.get("video"),
                    results.get("camera_id", "CAM_01"),
                    results.get("store_id", "STORE_001"),
                    results.get("total_frames"),
                    results.get("processed_frames"),
                    results.get("total_events"),
                    json.dumps(results.get("metrics", {}), default=str),
                ),
            )

    # ═══ Utilities ═══

    def clear_all(self):
        """Clear all data — use with caution."""
        with self._get_conn() as conn:
            conn.executescript("""
                DELETE FROM events;
                DELETE FROM journeys;
                DELETE FROM anomalies;
                DELETE FROM pos_transactions;
                DELETE FROM processing_runs;
            """)

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_conn() as conn:
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            journeys = conn.execute("SELECT COUNT(*) FROM journeys").fetchone()[0]
            anomalies = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM processing_runs").fetchone()[0]
            pos = conn.execute("SELECT COUNT(*) FROM pos_transactions").fetchone()[0]
            return {
                "total_events": events,
                "total_journeys": journeys,
                "total_anomalies": anomalies,
                "processing_runs": runs,
                "pos_transactions": pos,
            }
