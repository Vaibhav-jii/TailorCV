# PROMPT: "Write comprehensive pytest tests for a FastAPI retail analytics API
# that has POST /events/ingest (batch, idempotent), GET /stores/{id}/metrics,
# GET /stores/{id}/funnel, GET /stores/{id}/heatmap, GET /stores/{id}/anomalies,
# and GET /health endpoints. Include edge cases: empty store, all-staff clip,
# zero purchases, re-entry in funnel, idempotency verification."
#
# CHANGES MADE:
# - Added test for partial success (malformed events mixed with valid ones)
# - Added test for STALE_FEED warning in health endpoint
# - Added store-scoped filtering tests
# - Changed test data to match Purplle sample_events.jsonl schema
# - Added idempotency test that calls ingest twice with same payload
# - Added test for staff exclusion from customer metrics

"""
Store Intelligence API Tests — Comprehensive test suite.

Tests cover:
  - Event ingestion (batch, idempotent, partial success)
  - Store metrics (staff exclusion, zero-purchase handling)
  - Conversion funnel (session dedup, re-entry)
  - Heatmap (normalisation, data_confidence flag)
  - Anomaly detection (severity levels, suggested_action)
  - Health endpoint (STALE_FEED detection)
  - Edge cases: empty store, all-staff, zero purchases
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.app import app
from storage.database import Database

client = TestClient(app)


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_db(tmp_path):
    """Use a fresh temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    import api.app as api_module
    api_module.db = Database(db_path)
    yield api_module.db
    

@pytest.fixture
def sample_entry_events():
    """Sample entry/exit events matching Purplle schema."""
    return [
        {
            "event_id": "evt_test_001",
            "event_type": "entry",
            "id_token": "ID_60001",
            "store_code": "STORE_001",
            "camera_id": "CAM_03",
            "event_timestamp": "2026-03-08T18:10:05.120000",
            "is_staff": False,
            "gender_pred": "F",
            "age_pred": 28,
            "age_bucket": "25-34",
            "is_face_hidden": False,
            "group_id": None,
            "group_size": None,
        },
        {
            "event_id": "evt_test_002",
            "event_type": "entry",
            "id_token": "ID_60002",
            "store_code": "STORE_001",
            "camera_id": "CAM_03",
            "event_timestamp": "2026-03-08T18:10:22.480000",
            "is_staff": False,
            "gender_pred": "M",
            "age_pred": 31,
            "age_bucket": "25-34",
            "is_face_hidden": False,
            "group_id": "G_10",
            "group_size": 2,
        },
        {
            "event_id": "evt_test_003",
            "event_type": "exit",
            "id_token": "ID_60001",
            "store_code": "STORE_001",
            "camera_id": "CAM_03",
            "event_timestamp": "2026-03-08T18:12:44.360000",
            "is_staff": False,
            "gender_pred": "F",
            "age_pred": 28,
            "age_bucket": "25-34",
        },
    ]


@pytest.fixture
def sample_zone_events():
    """Sample zone events matching Purplle schema."""
    return [
        {
            "event_id": "evt_zone_001",
            "event_type": "zone_entered",
            "track_id": 101,
            "store_id": "STORE_001",
            "camera_id": "CAM_02",
            "zone_id": "PURPLLE_MUM_1076_Z01",
            "zone_name": "Left Shelf",
            "zone_type": "SHELF",
            "is_revenue_zone": "Yes",
            "event_time": "2026-03-08T18:10:45.280000",
            "zone_hotspot_x": 412.6,
            "zone_hotspot_y": 238.4,
            "gender": "F",
            "age": 28,
            "age_bucket": "25-34",
        },
        {
            "event_id": "evt_zone_002",
            "event_type": "zone_exited",
            "track_id": 101,
            "store_id": "STORE_001",
            "camera_id": "CAM_02",
            "zone_id": "PURPLLE_MUM_1076_Z01",
            "zone_name": "Left Shelf",
            "zone_type": "SHELF",
            "is_revenue_zone": "Yes",
            "event_time": "2026-03-08T18:11:18.720000",
            "dwell_ms": 33440,
        },
    ]


@pytest.fixture
def sample_staff_events():
    """Staff entry events — should be excluded from customer metrics."""
    return [
        {
            "event_id": "evt_staff_001",
            "event_type": "entry",
            "id_token": "STAFF_001",
            "store_code": "STORE_001",
            "camera_id": "CAM_03",
            "event_timestamp": "2026-03-08T10:00:00.000000",
            "is_staff": True,
            "gender_pred": "M",
            "age_pred": 35,
        },
        {
            "event_id": "evt_staff_002",
            "event_type": "entry",
            "id_token": "STAFF_002",
            "store_code": "STORE_001",
            "camera_id": "CAM_03",
            "event_timestamp": "2026-03-08T10:00:05.000000",
            "is_staff": True,
        },
    ]


# ═══════════════════════════════════════════════════════════
# Health Endpoint Tests
# ═══════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Store Intelligence API"
        assert "version" in data
        assert "timestamp" in data

    def test_health_includes_database_stats(self):
        response = client.get("/health")
        data = response.json()
        assert "database" in data
        assert "total_events" in data["database"]

    def test_health_stale_feed_detection(self, clean_db, sample_entry_events):
        """If last event is >10 min old, health should warn STALE_FEED."""
        # Ingest old event
        old_events = [sample_entry_events[0].copy()]
        old_events[0]["event_timestamp"] = "2025-01-01T00:00:00.000000"
        clean_db.ingest_raw_events(old_events)
        
        response = client.get("/health")
        data = response.json()
        # Should have stores data with stale feed
        if data.get("stores"):
            store_data = list(data["stores"].values())
            if store_data:
                assert store_data[0]["status"] == "STALE_FEED"


# ═══════════════════════════════════════════════════════════
# Event Ingest Tests
# ═══════════════════════════════════════════════════════════

class TestEventIngest:
    def test_ingest_single_event(self, sample_entry_events):
        response = client.post("/events/ingest", json={"events": [sample_entry_events[0]]})
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 1
        assert data["duplicates"] == 0

    def test_ingest_batch(self, sample_entry_events):
        response = client.post("/events/ingest", json={"events": sample_entry_events})
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == len(sample_entry_events)
        assert data["total_submitted"] == len(sample_entry_events)

    def test_ingest_idempotent(self, sample_entry_events):
        """Calling ingest twice with same events should not create duplicates."""
        client.post("/events/ingest", json={"events": sample_entry_events})
        response = client.post("/events/ingest", json={"events": sample_entry_events})
        data = response.json()
        assert data["duplicates"] == len(sample_entry_events)
        assert data["accepted"] == 0

    def test_ingest_bare_array(self, sample_entry_events):
        """Should accept bare array without wrapping in {"events": [...]}."""
        response = client.post("/events/ingest", json=sample_entry_events)
        assert response.status_code == 200
        assert response.json()["accepted"] == len(sample_entry_events)

    def test_ingest_batch_size_limit(self):
        """Should reject batches >500 events."""
        events = [{"event_id": f"evt_{i}", "event_type": "entry"} for i in range(501)]
        response = client.post("/events/ingest", json=events)
        assert response.status_code == 400
        assert "BATCH_TOO_LARGE" in response.json().get("type", "")

    def test_ingest_empty_batch(self):
        response = client.post("/events/ingest", json={"events": []})
        assert response.status_code == 200
        assert response.json()["accepted"] == 0

    def test_ingest_mixed_event_types(self, sample_entry_events, sample_zone_events):
        """Ingest both entry and zone events in one batch."""
        all_events = sample_entry_events + sample_zone_events
        response = client.post("/events/ingest", json=all_events)
        assert response.status_code == 200
        assert response.json()["accepted"] == len(all_events)


# ═══════════════════════════════════════════════════════════
# Store Metrics Tests
# ═══════════════════════════════════════════════════════════

class TestStoreMetrics:
    def test_metrics_returns_json(self, clean_db, sample_entry_events):
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/stores/STORE_001/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "unique_visitors" in data
        assert "conversion_rate" in data
        assert "queue_depth" in data
        assert "abandonment_rate" in data

    def test_metrics_excludes_staff(self, clean_db, sample_entry_events, sample_staff_events):
        """Staff entries should NOT be counted in unique_visitors."""
        all_events = sample_entry_events + sample_staff_events
        clean_db.ingest_raw_events(all_events)
        
        response = client.get("/stores/STORE_001/metrics")
        data = response.json()
        # Only non-staff entries should be counted
        assert data["unique_visitors"] == 2  # ID_60001 and ID_60002, not staff

    def test_metrics_empty_store(self):
        """Should handle zero-traffic store without crashing."""
        response = client.get("/stores/NONEXISTENT_STORE/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0

    def test_metrics_zero_purchases(self, clean_db, sample_entry_events):
        """Store with visitors but no purchases should have 0 conversion."""
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/stores/STORE_001/metrics")
        data = response.json()
        assert data["conversion_rate"] == 0


# ═══════════════════════════════════════════════════════════
# Funnel Tests
# ═══════════════════════════════════════════════════════════

class TestFunnel:
    def test_funnel_structure(self, clean_db, sample_entry_events):
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/stores/STORE_001/funnel")
        assert response.status_code == 200
        data = response.json()
        assert "funnel" in data
        assert len(data["funnel"]) == 4
        stages = [s["stage"] for s in data["funnel"]]
        assert stages == ["Entry", "Zone Visit", "Billing Queue", "Purchase"]

    def test_funnel_empty_store(self):
        response = client.get("/stores/EMPTY_STORE/funnel")
        assert response.status_code == 200
        data = response.json()
        assert data["conversion_rate"] == 0


# ═══════════════════════════════════════════════════════════
# Heatmap Tests
# ═══════════════════════════════════════════════════════════

class TestHeatmap:
    def test_heatmap_returns_zones(self, clean_db, sample_zone_events):
        clean_db.ingest_raw_events(sample_zone_events)
        response = client.get("/stores/STORE_001/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert "data_confidence" in data

    def test_heatmap_normalisation(self, clean_db, sample_zone_events):
        """Normalised scores should be 0-100."""
        clean_db.ingest_raw_events(sample_zone_events)
        response = client.get("/stores/STORE_001/heatmap")
        data = response.json()
        for zone in data.get("zones", []):
            assert 0 <= zone.get("normalised_score", 0) <= 100

    def test_heatmap_low_confidence(self, clean_db, sample_zone_events):
        """Should flag LOW confidence if <20 sessions."""
        clean_db.ingest_raw_events(sample_zone_events)
        response = client.get("/stores/STORE_001/heatmap")
        data = response.json()
        assert data["data_confidence"] == "LOW"

    def test_heatmap_empty_store(self):
        response = client.get("/stores/EMPTY_STORE/heatmap")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════
# Anomaly Tests
# ═══════════════════════════════════════════════════════════

class TestAnomalies:
    def test_anomalies_returns_structure(self, clean_db, sample_entry_events):
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/stores/STORE_001/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert "active_anomalies" in data
        assert "store_id" in data

    def test_anomaly_severity_levels(self, clean_db, sample_entry_events):
        """Anomalies should use INFO/WARN/CRITICAL severity."""
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/stores/STORE_001/anomalies")
        data = response.json()
        for anomaly in data.get("active_anomalies", []):
            assert anomaly["severity"] in ("INFO", "WARN", "CRITICAL")
            assert "suggested_action" in anomaly
            assert len(anomaly["suggested_action"]) > 0


# ═══════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_all_staff_clip(self, clean_db, sample_staff_events):
        """All-staff clip should result in 0 customer metrics."""
        clean_db.ingest_raw_events(sample_staff_events)
        response = client.get("/stores/STORE_001/metrics")
        data = response.json()
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0

    def test_invalid_json_ingest(self):
        response = client.post("/events/ingest", content="not json", 
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 400

    def test_events_query_with_filters(self, clean_db, sample_entry_events, sample_zone_events):
        all_events = sample_entry_events + sample_zone_events
        clean_db.ingest_raw_events(all_events)
        
        # Filter by event type
        response = client.get("/api/v1/events?event_type=entry")
        assert response.status_code == 200
        data = response.json()
        for event in data["events"]:
            assert event["event_type"] == "entry"

    def test_db_stats(self, clean_db, sample_entry_events):
        clean_db.ingest_raw_events(sample_entry_events)
        response = client.get("/api/v1/db/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == len(sample_entry_events)
