"""
StoreIQ Person Tracker — ByteTrack multi-object tracking.

ADR: ByteTrack chosen over DeepSORT because:
1. No separate re-ID model needed → simpler deployment, less compute
2. Better performance on crowded retail scenes
3. State-of-the-art MOT benchmarks with lower complexity
4. DeepSORT requires a feature extractor → more moving parts
"""

import supervision as sv
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


class PersonTracker:
    """
    Multi-object tracker using ByteTrack algorithm.
    
    Assigns persistent IDs to detected persons across frames.
    Handles short-term occlusions via Kalman filter prediction.
    """

    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=settings.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=settings.LOST_TRACK_BUFFER,
            minimum_matching_threshold=settings.MATCH_THRESHOLD,
            frame_rate=settings.FRAME_RATE,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections and return tracked detections.
        
        The returned detections will have `tracker_id` populated with
        persistent IDs that survive across frames.
        
        Args:
            detections: Raw detections from PersonDetector
            
        Returns:
            sv.Detections with tracker_id assigned
        """
        if len(detections) == 0:
            return detections

        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def reset(self):
        """Reset tracker state — use when switching to a new video."""
        self.tracker.reset()
