"""
Store Intelligence Data Models — Aligned to Purplle challenge schema.

Three event categories based on sample_events.jsonl:
  1. Entry/Exit events (entry camera)
  2. Zone events (zone cameras)
  3. Queue/Billing events (billing camera)

Each has slightly different fields but shares common base attributes.
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, List, Any
import uuid


# ═══════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════

class EventType(str, Enum):
    """All event types the system can generate."""
    ENTRY = "entry"
    EXIT = "exit"
    ZONE_ENTERED = "zone_entered"
    ZONE_EXITED = "zone_exited"
    ZONE_DWELL = "zone_dwell"
    QUEUE_JOINED = "queue_joined"
    QUEUE_COMPLETED = "queue_completed"
    QUEUE_ABANDONED = "queue_abandoned"
    REENTRY = "reentry"


class PersonState(str, Enum):
    """State machine states for a tracked person."""
    OUTSIDE = "OUTSIDE"
    IN_STORE = "IN_STORE"
    IN_ZONE = "IN_ZONE"
    IN_BILLING_QUEUE = "IN_BILLING_QUEUE"
    PURCHASED = "PURCHASED"
    EXITED = "EXITED"


# ═══════════════════════════════════════════════════════════
# Core Data Models
# ═══════════════════════════════════════════════════════════

class BoundingBox(BaseModel):
    """Bounding box coordinates (top-left corner + dimensions)."""
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def bottom_center(self) -> tuple:
        """Foot position — used for zone detection."""
        return (self.x + self.w / 2, self.y + self.h)


class StoreEvent(BaseModel):
    """
    Universal event model matching Purplle sample_events.jsonl schema.
    
    Supports three event sub-types:
    - Entry/exit: id_token, is_staff, gender/age, group info
    - Zone: track_id, zone_id, zone_name, zone_type, hotspot coords
    - Queue: queue_event_id, queue join/serve/exit timestamps, wait_seconds, abandoned
    """
    # Common fields
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    store_id: str = "STORE_001"
    camera_id: str = "CAM_01"
    timestamp: Optional[datetime] = None  # Primary timestamp — set by event engine
    
    # Entry/Exit event fields
    id_token: Optional[str] = None
    event_timestamp: Optional[str] = None
    is_staff: bool = False
    gender_pred: Optional[str] = None
    age_pred: Optional[int] = None
    age_bucket: Optional[str] = None
    is_face_hidden: bool = False
    group_id: Optional[str] = None
    group_size: Optional[int] = None
    
    # Zone event fields
    track_id: Optional[int] = None
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    zone_type: Optional[str] = None
    is_revenue_zone: Optional[str] = None
    event_time: Optional[str] = None
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    
    # Queue event fields
    queue_event_id: Optional[str] = None
    queue_join_ts: Optional[str] = None
    queue_served_ts: Optional[str] = None
    queue_exit_ts: Optional[str] = None
    wait_seconds: Optional[int] = None
    queue_position_at_join: Optional[int] = None
    abandoned: Optional[bool] = None
    
    # Internal tracking (not in output)
    confidence: float = 0.0
    frame_number: int = 0
    person_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Dwell tracking
    dwell_ms: Optional[int] = None
    session_seq: Optional[int] = None

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
    
    def to_output_dict(self) -> dict:
        """Convert to the output format matching sample_events.jsonl."""
        etype = self.event_type.value
        
        if etype in ("entry", "exit", "reentry"):
            return {
                "event_type": etype,
                "id_token": self.id_token or self.person_id,
                "store_code": self.store_id,
                "camera_id": self.camera_id,
                "event_timestamp": self.event_timestamp or self.event_time,
                "is_staff": self.is_staff,
                "gender_pred": self.gender_pred or self.gender,
                "age_pred": self.age_pred or self.age,
                "age_bucket": self.age_bucket,
                "is_face_hidden": self.is_face_hidden,
                "group_id": self.group_id,
                "group_size": self.group_size,
            }
        elif etype in ("zone_entered", "zone_exited", "zone_dwell"):
            return {
                "event_type": etype,
                "track_id": self.track_id,
                "store_id": self.store_id,
                "camera_id": self.camera_id,
                "zone_id": self.zone_id,
                "zone_name": self.zone_name,
                "zone_type": self.zone_type,
                "is_revenue_zone": self.is_revenue_zone or "Yes",
                "event_time": self.event_time or self.event_timestamp,
                "zone_hotspot_x": self.zone_hotspot_x,
                "zone_hotspot_y": self.zone_hotspot_y,
                "gender": self.gender or self.gender_pred,
                "age": self.age or self.age_pred,
                "age_bucket": self.age_bucket,
                "dwell_ms": self.dwell_ms,
            }
        elif etype in ("queue_joined", "queue_completed", "queue_abandoned"):
            return {
                "queue_event_id": self.queue_event_id or str(uuid.uuid4()),
                "event_type": etype,
                "track_id": self.track_id,
                "store_id": self.store_id,
                "camera_id": self.camera_id,
                "zone_id": self.zone_id,
                "zone_name": self.zone_name or "Billing Counter Queue",
                "zone_type": self.zone_type or "BILLING",
                "is_revenue_zone": self.is_revenue_zone or "Yes",
                "queue_join_ts": self.queue_join_ts,
                "queue_served_ts": self.queue_served_ts,
                "queue_exit_ts": self.queue_exit_ts,
                "wait_seconds": self.wait_seconds,
                "queue_position_at_join": self.queue_position_at_join,
                "abandoned": self.abandoned,
                "zone_hotspot_x": self.zone_hotspot_x,
                "zone_hotspot_y": self.zone_hotspot_y,
                "gender": self.gender or self.gender_pred,
                "age": self.age or self.age_pred,
                "age_bucket": self.age_bucket,
            }
        else:
            return {"event_type": etype, **self.metadata}


class PersonJourney(BaseModel):
    """
    Complete journey of a single person through the store.
    Reconstructed from individual events after the person exits.
    """
    person_id: str
    track_id: int
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    zones_visited: List[Dict[str, Any]] = Field(default_factory=list)
    total_dwell_time: float = 0.0
    reached_billing: bool = False
    purchased: bool = False
    abandoned_billing: bool = False
    is_reentry: bool = False
    is_staff: bool = False
    event_count: int = 0
    gender: Optional[str] = None
    age: Optional[int] = None
    age_bucket: Optional[str] = None
    group_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Person Tracking State (Mutable — used by Event Engine)
# ═══════════════════════════════════════════════════════════

class PersonTrackState:
    """
    Mutable runtime state for a tracked person.
    
    This is NOT a Pydantic model — it's a stateful object maintained
    by the EventEngine during video processing. Converted to a
    PersonJourney when the person exits.
    """

    def __init__(self, track_id: int, first_seen_frame: int, first_seen_time: datetime):
        self.track_id: int = track_id
        self.person_id: str = f"ID_{track_id:05d}"
        self.state: PersonState = PersonState.OUTSIDE

        # Temporal tracking
        self.first_seen_frame: int = first_seen_frame
        self.first_seen_time: datetime = first_seen_time
        self.last_seen_frame: int = first_seen_frame
        self.last_seen_time: datetime = first_seen_time

        # Zone tracking
        self.current_zone: Optional[str] = None
        self.current_zone_name: Optional[str] = None
        self.current_zone_type: Optional[str] = None
        self.zone_enter_time: Optional[datetime] = None
        self.zone_enter_frame: Optional[int] = None
        self.zones_visited: List[Dict[str, Any]] = []
        self.last_dwell_emit_time: Optional[datetime] = None  # For 30s dwell emission

        # Journey flags
        self.has_entered_store: bool = False
        self.has_exited_store: bool = False
        self.reached_billing: bool = False
        self.purchased: bool = False
        self.abandoned_billing: bool = False
        self.is_reentry: bool = False
        self.is_staff: bool = False
        
        # Demographics
        self.gender: Optional[str] = None
        self.age: Optional[int] = None
        self.age_bucket: Optional[str] = None
        
        # Group tracking
        self.group_id: Optional[str] = None
        self.group_size: Optional[int] = None

        # Detection data
        self.last_bbox: Optional[BoundingBox] = None
        self.events: List[StoreEvent] = []
        self.session_seq: int = 0
        
        # Staff detection heuristics
        self.total_frames_seen: int = 0
        self.zone_change_count: int = 0
        self.zones_entered: set = set()

    def next_seq(self) -> int:
        """Get next session sequence number."""
        self.session_seq += 1
        return self.session_seq

    def add_event(self, event: StoreEvent):
        """Record an event for this person."""
        self.events.append(event)

    def to_journey(self) -> PersonJourney:
        """Convert mutable state to an immutable PersonJourney record."""
        return PersonJourney(
            person_id=self.person_id,
            track_id=self.track_id,
            entry_time=self.first_seen_time,
            exit_time=self.last_seen_time if self.has_exited_store else None,
            zones_visited=self.zones_visited,
            total_dwell_time=sum(
                z.get("dwell_seconds", 0) for z in self.zones_visited
            ),
            reached_billing=self.reached_billing,
            purchased=self.purchased,
            abandoned_billing=self.abandoned_billing,
            is_reentry=self.is_reentry,
            is_staff=self.is_staff,
            event_count=len(self.events),
            gender=self.gender,
            age=self.age,
            age_bucket=self.age_bucket,
            group_id=self.group_id,
        )


# ═══════════════════════════════════════════════════════════
# Analytics Models
# ═══════════════════════════════════════════════════════════

class ZoneMetrics(BaseModel):
    """Analytics for a single store zone."""
    zone_name: str
    zone_type: str
    current_occupancy: int = 0
    total_visits: int = 0
    unique_visitors: int = 0
    avg_dwell_seconds: float = 0.0
    max_dwell_seconds: float = 0.0


class StoreMetrics(BaseModel):
    """Aggregate store-level metrics."""
    total_entries: int = 0
    total_exits: int = 0
    current_occupancy: int = 0
    total_unique_visitors: int = 0
    conversion_rate: float = 0.0
    avg_visit_duration_seconds: float = 0.0
    billing_reach_rate: float = 0.0
    billing_abandon_rate: float = 0.0
    zone_metrics: Dict[str, ZoneMetrics] = Field(default_factory=dict)


class ConversionFunnel(BaseModel):
    """Conversion funnel metrics — the North Star."""
    total_entries: int = 0
    browsed_zones: int = 0
    reached_billing: int = 0
    purchased: int = 0
    conversion_rate: float = 0.0
    billing_reach_rate: float = 0.0
    browse_rate: float = 0.0


class AnomalyAlert(BaseModel):
    """A detected anomaly with severity and context."""
    alert_id: str = Field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    timestamp: datetime
    anomaly_type: str
    severity: str  # "INFO", "WARN", "CRITICAL"
    metric_name: str
    current_value: float
    expected_range: str
    message: str
    suggested_action: str = ""
    store_id: str = ""
