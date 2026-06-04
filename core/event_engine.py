"""
StoreIQ Event Engine — The Brain of the System.

Maintains a per-person state machine and generates structured events
based on zone transitions, dwell times, and behavioral patterns.

State Machine Per Person:
    OUTSIDE → ENTRY → IN_STORE → ZONE_ENTER → IN_ZONE → ZONE_EXIT
           → BILLING_QUEUE_JOIN → PURCHASE_INFERRED / BILLING_QUEUE_ABANDON
           → EXIT

This is the most critical component — it transforms raw detections
into business-meaningful events, aligned to the Purplle challenge schema.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import supervision as sv
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    EventType, PersonTrackState, PersonJourney,
    StoreEvent, BoundingBox, PersonState
)
from core.zone_manager import ZoneManager
from config.settings import settings


class EventEngine:
    """
    The brain of StoreIQ.
    
    Maintains per-person state machines and generates structured events
    based on zone transitions and behavioral patterns.
    """

    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager
        self.active_tracks: Dict[int, PersonTrackState] = {}
        self.completed_journeys: List[PersonJourney] = []
        self.exited_persons: List[PersonTrackState] = []
        self.all_events: List[StoreEvent] = []
        self.frame_count: int = 0
        self._event_count: int = 0

    # ─── Main Processing Loop ────────────────────────────

    def process_frame(
        self,
        detections: sv.Detections,
        frame_number: int,
        timestamp: datetime,
        camera_id: str = "CAM_01",
        store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        self.frame_count = frame_number
        frame_events: List[StoreEvent] = []

        if detections.tracker_id is None or len(detections) == 0:
            lost_events = self._check_lost_tracks(frame_number, timestamp, set(), camera_id, store_id)
            self.all_events.extend(lost_events)
            return lost_events

        zone_results = self.zone_manager.check_zones(detections)
        current_track_ids: set = set()

        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i])
            current_track_ids.add(track_id)

            bbox = BoundingBox(
                x=float(detections.xyxy[i][0]),
                y=float(detections.xyxy[i][1]),
                w=float(detections.xyxy[i][2] - detections.xyxy[i][0]),
                h=float(detections.xyxy[i][3] - detections.xyxy[i][1]),
            )
            confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0

            # ── Handle new person ──
            if track_id not in self.active_tracks:
                new_events = self._handle_new_person(
                    track_id, frame_number, timestamp, bbox, confidence,
                    camera_id, store_id,
                )
                frame_events.extend(new_events)

            person = self.active_tracks[track_id]
            person.last_seen_frame = frame_number
            person.last_seen_time = timestamp
            person.last_bbox = bbox
            person.total_frames_seen += 1
            
            # Group detection (simple spatial proximity heuristic)
            self._update_group_assignment(person, detections, i)

            # ── Determine current zone ──
            detected_zone: Optional[str] = None
            for zone_name, mask in zone_results.items():
                if mask[i]:
                    detected_zone = zone_name
                    break

            # ── Staff detection heuristic update ──
            if detected_zone:
                person.zones_entered.add(detected_zone)
            self._update_staff_heuristic(person)

            # ── Handle zone transitions ──
            if detected_zone != person.current_zone:
                transition_events = self._handle_zone_change(
                    person, detected_zone, frame_number, timestamp,
                    bbox, confidence, camera_id, store_id,
                )
                frame_events.extend(transition_events)
            elif person.current_zone:
                # ── Handle periodic dwell emission (every 30s) ──
                dwell_events = self._check_dwell_emission(
                    person, frame_number, timestamp, bbox, confidence, camera_id, store_id
                )
                frame_events.extend(dwell_events)

        # ── Check for lost tracks ──
        lost_events = self._check_lost_tracks(
            frame_number, timestamp, current_track_ids,
            camera_id, store_id,
        )
        frame_events.extend(lost_events)

        self.all_events.extend(frame_events)
        return frame_events

    # ─── Staff Detection ──────────────────────────────

    def _update_staff_heuristic(self, person: PersonTrackState):
        """
        Staff detection heuristic:
        - Visible for many frames (>1000)
        - Visited many zones (>3)
        - High zone transition frequency
        """
        # A simple mock behavior for demonstration: 
        # If someone has been around a very long time and moving a lot, they are staff.
        if person.total_frames_seen > 150 and person.zone_change_count > 3:
            person.is_staff = True

    # ─── Group Detection ──────────────────────────────

    def _update_group_assignment(self, person: PersonTrackState, detections: sv.Detections, idx: int):
        """Simple spatial proximity group assignment."""
        if person.group_id:
            return  # Already assigned
            
        x1, y1 = float(detections.xyxy[idx][0]), float(detections.xyxy[idx][1])
        x2, y2 = float(detections.xyxy[idx][2]), float(detections.xyxy[idx][3])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        group_members = [person]
        
        for j in range(len(detections)):
            if idx == j:
                continue
            other_id = int(detections.tracker_id[j])
            if other_id in self.active_tracks:
                other_person = self.active_tracks[other_id]
                
                ox1, oy1 = float(detections.xyxy[j][0]), float(detections.xyxy[j][1])
                ox2, oy2 = float(detections.xyxy[j][2]), float(detections.xyxy[j][3])
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                
                dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
                if dist < 150:  # 150px proximity threshold
                    group_members.append(other_person)
                    
        if len(group_members) > 1:
            group_id = f"G_{person.track_id}"
            # Find if any already have a group
            for member in group_members:
                if member.group_id:
                    group_id = member.group_id
                    break
            for member in group_members:
                member.group_id = group_id
                member.group_size = len(group_members)

    # ─── New Person Handler ──────────────────────────────

    def _handle_new_person(
        self, track_id: int, frame_number: int, timestamp: datetime,
        bbox: BoundingBox, confidence: float,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []

        is_reentry, original_person_id = self._check_reentry(bbox, timestamp)

        person = PersonTrackState(track_id, frame_number, timestamp)
        person.last_bbox = bbox
        
        # Mock demographics for demo
        person.gender = random.choice(["M", "F"])
        person.age = random.randint(18, 65)
        if person.age < 25:
            person.age_bucket = "18-24"
        elif person.age < 35:
            person.age_bucket = "25-34"
        elif person.age < 45:
            person.age_bucket = "35-44"
        else:
            person.age_bucket = "45+"

        if is_reentry and original_person_id:
            person.is_reentry = True
            person.person_id = original_person_id

            event = StoreEvent(
                event_type=EventType.REENTRY,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                id_token=person.person_id,
                confidence=confidence,
                bbox=bbox,
                is_staff=person.is_staff,
                gender_pred=person.gender,
                age_pred=person.age,
                age_bucket=person.age_bucket,
                group_id=person.group_id,
                group_size=person.group_size,
                session_seq=person.next_seq(),
            )
            person.add_event(event)
            events.append(event)
        else:
            person.has_entered_store = True
            person.state = PersonState.IN_STORE

            event = StoreEvent(
                event_type=EventType.ENTRY,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                id_token=person.person_id,
                confidence=confidence,
                bbox=bbox,
                is_staff=person.is_staff,
                gender_pred=person.gender,
                age_pred=person.age,
                age_bucket=person.age_bucket,
                group_id=person.group_id,
                group_size=person.group_size,
                session_seq=person.next_seq(),
            )
            person.add_event(event)
            events.append(event)

        self.active_tracks[track_id] = person
        return events

    # ─── Periodic Dwell Handler ─────────────────────────────

    def _check_dwell_emission(
        self, person: PersonTrackState, frame_number: int, timestamp: datetime,
        bbox: BoundingBox, confidence: float,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []
        if not person.current_zone or not person.zone_enter_time:
            return events
            
        last_emit = person.last_dwell_emit_time or person.zone_enter_time
        dwell_since_last_emit = (timestamp - last_emit).total_seconds()
        
        # Emit every 30 seconds
        if dwell_since_last_emit >= 30.0:
            total_dwell = (timestamp - person.zone_enter_time).total_seconds()
            
            zone_id = f"Z_{person.current_zone.upper().replace(' ', '_')}"
            zone_type = self.zone_manager.get_zone_type(person.current_zone)
            
            event = StoreEvent(
                event_type=EventType.ZONE_DWELL,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                track_id=person.track_id,
                zone_id=zone_id,
                zone_name=person.current_zone,
                zone_type=zone_type,
                confidence=confidence,
                bbox=bbox,
                zone_hotspot_x=round(bbox.bottom_center[0], 1),
                zone_hotspot_y=round(bbox.bottom_center[1], 1),
                dwell_ms=int(total_dwell * 1000),
                gender=person.gender,
                age=person.age,
                age_bucket=person.age_bucket,
                session_seq=person.next_seq(),
            )
            person.add_event(event)
            events.append(event)
            person.last_dwell_emit_time = timestamp
            
        return events

    # ─── Zone Change Handler ─────────────────────────────

    def _handle_zone_change(
        self, person: PersonTrackState, new_zone: Optional[str],
        frame_number: int, timestamp: datetime,
        bbox: BoundingBox, confidence: float,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []
        person.zone_change_count += 1

        if person.current_zone is not None:
            exit_events = self._emit_zone_exit(
                person, person.current_zone, frame_number, timestamp,
                bbox, confidence, camera_id, store_id,
            )
            events.extend(exit_events)

        if new_zone is not None:
            enter_events = self._emit_zone_enter(
                person, new_zone, frame_number, timestamp,
                bbox, confidence, camera_id, store_id,
            )
            events.extend(enter_events)

        return events

    def _emit_zone_enter(
        self, person: PersonTrackState, zone_name: str,
        frame_number: int, timestamp: datetime,
        bbox: BoundingBox, confidence: float,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []

        person.current_zone = zone_name
        person.zone_enter_time = timestamp
        person.zone_enter_frame = frame_number
        person.last_dwell_emit_time = None
        
        zone_id = f"Z_{zone_name.upper().replace(' ', '_')}"
        zone_type = self.zone_manager.get_zone_type(zone_name)

        if zone_type == "billing":
            person.reached_billing = True
            
            # Simple queue depth calculation: count active tracks in billing zone
            queue_depth = sum(
                1 for p in self.active_tracks.values()
                if p.current_zone_type == "billing" or self.zone_manager.get_zone_type(p.current_zone) == "billing"
            )
            
            event = StoreEvent(
                event_type=EventType.QUEUE_JOINED,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                track_id=person.track_id,
                zone_id=zone_id,
                zone_name=zone_name,
                zone_type=zone_type,
                confidence=confidence,
                bbox=bbox,
                queue_join_ts=timestamp.isoformat(),
                queue_position_at_join=queue_depth,
                zone_hotspot_x=round(bbox.bottom_center[0], 1),
                zone_hotspot_y=round(bbox.bottom_center[1], 1),
                gender=person.gender,
                age=person.age,
                age_bucket=person.age_bucket,
                session_seq=person.next_seq(),
            )
        else:
            event = StoreEvent(
                event_type=EventType.ZONE_ENTERED,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                track_id=person.track_id,
                zone_id=zone_id,
                zone_name=zone_name,
                zone_type=zone_type,
                confidence=confidence,
                bbox=bbox,
                zone_hotspot_x=round(bbox.bottom_center[0], 1),
                zone_hotspot_y=round(bbox.bottom_center[1], 1),
                gender=person.gender,
                age=person.age,
                age_bucket=person.age_bucket,
                session_seq=person.next_seq(),
            )

        person.add_event(event)
        events.append(event)
        return events

    def _emit_zone_exit(
        self, person: PersonTrackState, zone_name: str,
        frame_number: int, timestamp: datetime,
        bbox: BoundingBox, confidence: float,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []
        dwell_seconds = 0.0
        if person.zone_enter_time:
            dwell_seconds = (timestamp - person.zone_enter_time).total_seconds()

        zone_id = f"Z_{zone_name.upper().replace(' ', '_')}"
        zone_type = self.zone_manager.get_zone_type(zone_name)

        person.zones_visited.append({
            "zone": zone_name,
            "zone_type": zone_type,
            "enter_time": person.zone_enter_time.isoformat() if person.zone_enter_time else None,
            "exit_time": timestamp.isoformat(),
            "dwell_seconds": round(dwell_seconds, 2),
            "enter_frame": person.zone_enter_frame,
            "exit_frame": frame_number,
        })

        if zone_type == "billing":
            if dwell_seconds >= settings.QUEUE_THRESHOLD_SECONDS:
                person.purchased = True
                event = StoreEvent(
                    event_type=EventType.QUEUE_COMPLETED,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    camera_id=camera_id,
                    store_id=store_id,
                    person_id=person.person_id,
                    track_id=person.track_id,
                    zone_id=zone_id,
                    zone_name=zone_name,
                    zone_type=zone_type,
                    confidence=min(confidence, 0.7),
                    bbox=bbox,
                    queue_join_ts=person.zone_enter_time.isoformat() if person.zone_enter_time else None,
                    queue_served_ts=(timestamp - timedelta(seconds=10)).isoformat(), # mock served ts
                    queue_exit_ts=timestamp.isoformat(),
                    wait_seconds=int(dwell_seconds),
                    abandoned=False,
                    zone_hotspot_x=round(bbox.bottom_center[0], 1),
                    zone_hotspot_y=round(bbox.bottom_center[1], 1),
                    gender=person.gender,
                    age=person.age,
                    age_bucket=person.age_bucket,
                    session_seq=person.next_seq(),
                )
            else:
                person.abandoned_billing = True
                event = StoreEvent(
                    event_type=EventType.QUEUE_ABANDONED,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    camera_id=camera_id,
                    store_id=store_id,
                    person_id=person.person_id,
                    track_id=person.track_id,
                    zone_id=zone_id,
                    zone_name=zone_name,
                    zone_type=zone_type,
                    confidence=confidence,
                    bbox=bbox,
                    queue_join_ts=person.zone_enter_time.isoformat() if person.zone_enter_time else None,
                    queue_exit_ts=timestamp.isoformat(),
                    wait_seconds=int(dwell_seconds),
                    abandoned=True,
                    zone_hotspot_x=round(bbox.bottom_center[0], 1),
                    zone_hotspot_y=round(bbox.bottom_center[1], 1),
                    gender=person.gender,
                    age=person.age,
                    age_bucket=person.age_bucket,
                    session_seq=person.next_seq(),
                )
        else:
            event = StoreEvent(
                event_type=EventType.ZONE_EXITED,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=camera_id,
                store_id=store_id,
                person_id=person.person_id,
                track_id=person.track_id,
                zone_id=zone_id,
                zone_name=zone_name,
                zone_type=zone_type,
                confidence=confidence,
                bbox=bbox,
                dwell_ms=int(dwell_seconds * 1000),
                zone_hotspot_x=round(bbox.bottom_center[0], 1),
                zone_hotspot_y=round(bbox.bottom_center[1], 1),
                gender=person.gender,
                age=person.age,
                age_bucket=person.age_bucket,
                session_seq=person.next_seq(),
            )
            
        person.add_event(event)
        events.append(event)

        person.current_zone = None
        person.zone_enter_time = None
        person.zone_enter_frame = None
        person.last_dwell_emit_time = None

        return events

    # ─── Lost Track Handler ──────────────────────────────

    def _check_lost_tracks(
        self, frame_number: int, timestamp: datetime,
        current_track_ids: set,
        camera_id: str = "CAM_01", store_id: str = "STORE_001",
    ) -> List[StoreEvent]:
        events = []
        lost_ids: List[int] = []

        for track_id, person in self.active_tracks.items():
            if track_id not in current_track_ids:
                frames_since_seen = frame_number - person.last_seen_frame

                if frames_since_seen > settings.LOST_TRACK_TIMEOUT_FRAMES:
                    if person.current_zone:
                        zone_exit_events = self._emit_zone_exit(
                            person, person.current_zone, frame_number, timestamp,
                            person.last_bbox or BoundingBox(x=0, y=0, w=0, h=0),
                            0.0, camera_id, store_id,
                        )
                        events.extend(zone_exit_events)

                    person.has_exited_store = True
                    exit_event = StoreEvent(
                        event_type=EventType.EXIT,
                        timestamp=person.last_seen_time,
                        frame_number=person.last_seen_frame,
                        camera_id=camera_id,
                        store_id=store_id,
                        person_id=person.person_id,
                        id_token=person.person_id,
                        bbox=person.last_bbox,
                        is_staff=person.is_staff,
                        gender_pred=person.gender,
                        age_pred=person.age,
                        age_bucket=person.age_bucket,
                        group_id=person.group_id,
                        group_size=person.group_size,
                        session_seq=person.next_seq(),
                        metadata={
                            "visit_duration_seconds": round(
                                (person.last_seen_time - person.first_seen_time).total_seconds(), 2
                            ),
                            "zones_visited_count": len(person.zones_visited),
                        },
                    )
                    person.add_event(exit_event)
                    events.append(exit_event)

                    self.exited_persons.append(person)
                    self.completed_journeys.append(person.to_journey())
                    lost_ids.append(track_id)

        for tid in lost_ids:
            del self.active_tracks[tid]

        return events

    # ─── Re-entry Detection ──────────────────────────────

    def _check_reentry(
        self, bbox: BoundingBox, timestamp: datetime
    ) -> Tuple[bool, Optional[str]]:
        if not self.exited_persons:
            return False, None

        for exited in reversed(self.exited_persons[-20:]):
            time_since_exit = (timestamp - exited.last_seen_time).total_seconds()
            if time_since_exit > settings.REENTRY_WINDOW_SECONDS:
                continue

            if exited.last_bbox:
                dx = abs(bbox.x - exited.last_bbox.x)
                dy = abs(bbox.y - exited.last_bbox.y)
                if dx < 150 and dy < 150:
                    return True, exited.person_id

        return False, None

    # ─── Metrics & Accessors ─────────────────────────────

    def get_active_count(self) -> int:
        return len(self.active_tracks)

    def get_all_journeys(self) -> List[PersonJourney]:
        journeys = list(self.completed_journeys)
        for person in self.active_tracks.values():
            journeys.append(person.to_journey())
        return journeys

    def finalize(self, frame_number: int, timestamp: datetime) -> List[StoreEvent]:
        """Force exit for all active tracks at end of video."""
        events = self._check_lost_tracks(
            frame_number + settings.LOST_TRACK_TIMEOUT_FRAMES + 1,
            timestamp,
            set(),
        )
        self.all_events.extend(events)
        return events

    def get_store_metrics(self) -> dict:
        total_entries = sum(1 for j in self.completed_journeys if j.entry_time) + \
                        sum(1 for p in self.active_tracks.values() if p.has_entered_store)
        
        total_exits = sum(1 for j in self.completed_journeys if j.exit_time)
        reached_billing = sum(1 for j in self.get_all_journeys() if j.reached_billing)
        purchased = sum(1 for j in self.get_all_journeys() if j.purchased)
        
        conversion_rate = (purchased / total_entries) if total_entries > 0 else 0.0
        billing_reach_rate = (reached_billing / total_entries) if total_entries > 0 else 0.0
        
        completed = [j for j in self.completed_journeys if j.entry_time and j.exit_time]
        avg_visit_duration = (
            sum((j.exit_time - j.entry_time).total_seconds() for j in completed) / len(completed)
        ) if completed else 0.0
        
        browsed_zones = sum(1 for j in self.get_all_journeys() if any(z.get("zone_type") != "ENTRANCE" for z in j.zones_visited))

        return {
            "total_entries": total_entries,
            "total_exits": total_exits,
            "current_occupancy": self.get_active_count(),
            "reached_billing": reached_billing,
            "purchased": purchased,
            "conversion_rate": conversion_rate,
            "billing_reach_rate": billing_reach_rate,
            "avg_visit_duration": avg_visit_duration,
            "browsed_zones": browsed_zones,
        }

    def get_zone_occupancy(self) -> dict:
        occ = {}
        for person in self.active_tracks.values():
            if person.current_zone:
                occ[person.current_zone] = occ.get(person.current_zone, 0) + 1
        return occ
