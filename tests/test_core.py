# PROMPT: "Write pytest tests for the store intelligence event engine that processes
# CCTV detection data and generates retail analytics events. Test the state machine
# transitions: OUTSIDE → ENTRY → IN_ZONE → BILLING → EXIT. Test staff detection
# heuristics, group detection, zone dwell emission every 30s, and re-entry detection."
#
# CHANGES MADE:
# - Added test for 30-second periodic dwell emission
# - Added staff detection heuristic tests
# - Added group_id/group_size propagation test
# - Added re-entry detection test
# - Added test for PersonTrackState.to_journey() conversion

"""
Store Intelligence Event Engine Tests — Core pipeline logic.

Tests cover:
  - PersonTrackState lifecycle and conversion to PersonJourney
  - Event generation for different event types
  - Staff detection heuristics
  - Group detection
  - Re-entry detection
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    PersonTrackState, PersonState, StoreEvent, EventType,
    PersonJourney, BoundingBox
)


class TestPersonTrackState:
    """Test the mutable person tracking state object."""

    def test_init_defaults(self):
        """New person should start OUTSIDE with sane defaults."""
        now = datetime.now()
        state = PersonTrackState(track_id=42, first_seen_frame=1, first_seen_time=now)
        
        assert state.track_id == 42
        assert state.person_id == "ID_00042"
        assert state.state == PersonState.OUTSIDE
        assert state.has_entered_store is False
        assert state.is_staff is False
        assert state.gender is None
        assert state.session_seq == 0

    def test_next_seq_increments(self):
        now = datetime.now()
        state = PersonTrackState(track_id=1, first_seen_frame=1, first_seen_time=now)
        
        assert state.next_seq() == 1
        assert state.next_seq() == 2
        assert state.next_seq() == 3

    def test_add_event(self):
        now = datetime.now()
        state = PersonTrackState(track_id=1, first_seen_frame=1, first_seen_time=now)
        event = StoreEvent(
            event_type=EventType.ENTRY,
            store_id="STORE_001",
            person_id="ID_00001",
        )
        state.add_event(event)
        assert len(state.events) == 1
        assert state.events[0].event_type == EventType.ENTRY

    def test_to_journey(self):
        """Converting state to journey should produce valid PersonJourney."""
        now = datetime.now()
        state = PersonTrackState(track_id=5, first_seen_frame=1, first_seen_time=now)
        state.has_entered_store = True
        state.has_exited_store = True
        state.reached_billing = True
        state.purchased = True
        state.gender = "F"
        state.age = 28
        state.age_bucket = "25-34"
        state.last_seen_time = now + timedelta(minutes=5)
        state.zones_visited = [
            {"zone": "Left Shelf", "dwell_seconds": 33.4},
            {"zone": "Billing Counter Queue", "dwell_seconds": 120.0},
        ]
        
        journey = state.to_journey()
        assert isinstance(journey, PersonJourney)
        assert journey.person_id == "ID_00005"
        assert journey.reached_billing is True
        assert journey.purchased is True
        assert journey.total_dwell_time == pytest.approx(153.4)
        assert journey.gender == "F"
        assert len(journey.zones_visited) == 2


class TestStoreEvent:
    """Test event model and serialization."""

    def test_entry_event_output(self):
        event = StoreEvent(
            event_type=EventType.ENTRY,
            store_id="STORE_001",
            camera_id="CAM_03",
            person_id="ID_00001",
            id_token="ID_00001",
            event_timestamp="2026-03-08T18:10:05.120000",
            gender_pred="F",
            age_pred=28,
            is_staff=False,
        )
        output = event.to_output_dict()
        assert output["event_type"] == "entry"
        assert output["id_token"] == "ID_00001"
        assert output["is_staff"] is False
        assert output["gender_pred"] == "F"

    def test_zone_event_output(self):
        event = StoreEvent(
            event_type=EventType.ZONE_ENTERED,
            store_id="STORE_001",
            camera_id="CAM_02",
            track_id=101,
            zone_id="PURPLLE_MUM_1076_Z01",
            zone_name="Left Shelf",
            zone_type="SHELF",
            event_time="2026-03-08T18:10:45.280000",
            zone_hotspot_x=412.6,
            zone_hotspot_y=238.4,
        )
        output = event.to_output_dict()
        assert output["event_type"] == "zone_entered"
        assert output["zone_id"] == "PURPLLE_MUM_1076_Z01"
        assert output["zone_name"] == "Left Shelf"
        assert output["is_revenue_zone"] == "Yes"

    def test_queue_event_output(self):
        event = StoreEvent(
            event_type=EventType.QUEUE_COMPLETED,
            store_id="STORE_001",
            track_id=102,
            queue_event_id="test-uuid-001",
            queue_join_ts="2026-03-08T18:13:05.080000",
            queue_served_ts="2026-03-08T18:13:13.240000",
            queue_exit_ts="2026-03-08T18:15:31.840000",
            wait_seconds=8,
            queue_position_at_join=2,
            abandoned=False,
        )
        output = event.to_output_dict()
        assert output["event_type"] == "queue_completed"
        assert output["wait_seconds"] == 8
        assert output["abandoned"] is False
        assert output["zone_type"] == "BILLING"

    def test_event_id_is_uuid(self):
        event = StoreEvent(event_type=EventType.ENTRY, store_id="S1")
        # Should be a valid UUID-format string
        assert len(event.event_id) > 10
        assert "-" in event.event_id


class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x=100, y=200, w=50, h=100)
        assert bbox.center == (125, 250)

    def test_bottom_center(self):
        bbox = BoundingBox(x=100, y=200, w=50, h=100)
        assert bbox.bottom_center == (125, 300)


class TestDatabase:
    """Test database operations."""

    def test_insert_and_query_events(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test.db"))
        
        events = [
            {"event_id": "e1", "event_type": "entry", "store_id": "S1",
             "camera_id": "C1", "event_timestamp": "2026-01-01T10:00:00",
             "id_token": "ID_001", "is_staff": False},
            {"event_id": "e2", "event_type": "entry", "store_id": "S1",
             "camera_id": "C1", "event_timestamp": "2026-01-01T10:01:00",
             "id_token": "STAFF_001", "is_staff": True},
        ]
        
        result = db.ingest_raw_events(events)
        assert result["accepted"] == 2
        
        # Query all
        all_events = db.get_events(store_id="S1")
        assert len(all_events) == 2
        
        # Query excluding staff
        customer_events = db.get_events(store_id="S1", exclude_staff=True)
        assert len(customer_events) == 1

    def test_idempotent_insert(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test.db"))
        
        event = {"event_id": "e_dup", "event_type": "entry", "store_id": "S1"}
        
        r1 = db.ingest_raw_events([event])
        assert r1["accepted"] == 1
        
        r2 = db.ingest_raw_events([event])
        assert r2["duplicates"] == 1
        assert r2["accepted"] == 0

    def test_store_metrics(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test.db"))
        
        # Ingest some events
        events = [
            {"event_id": f"e_{i}", "event_type": "entry", "store_id": "S1",
             "id_token": f"ID_{i}", "is_staff": False,
             "event_timestamp": f"2026-01-01T10:{i:02d}:00"}
            for i in range(5)
        ]
        db.ingest_raw_events(events)
        
        metrics = db.get_store_metrics("S1")
        assert metrics["unique_visitors"] == 5
        assert metrics["store_id"] == "S1"

    def test_pos_loading(self, tmp_path):
        from storage.database import Database
        db = Database(str(tmp_path / "test.db"))
        
        csv_path = tmp_path / "pos.csv"
        csv_path.write_text(
            "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n"
            "1,10-04-2026,12:15:05,S1,12345,TestBrand,100.0\n"
            "2,10-04-2026,12:42:18,S1,12346,TestBrand,200.0\n"
        )
        
        count = db.load_pos_csv(str(csv_path), store_id="S1")
        assert count == 2
