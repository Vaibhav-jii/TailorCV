"""
StoreIQ Person Detector — YOLOv8 wrapper for person detection.

Uses Ultralytics YOLOv8 with supervision library integration.
Only detects class 0 (person) — all other classes are filtered out.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


class PersonDetector:
    """
    Wraps YOLOv8 for person-only detection.
    
    ADR: YOLOv8 chosen over RT-DETR for better speed/accuracy trade-off
    at retail-store scale. RT-DETR has better accuracy on small objects
    but 3x slower inference — unacceptable for real-time processing.
    """

    def __init__(self, model_path: str = None, confidence: float = None, iou: float = None):
        model_name = model_path or settings.YOLO_MODEL
        self.model = YOLO(model_name)
        self.confidence = confidence or settings.CONFIDENCE_THRESHOLD
        self.iou = iou or settings.IOU_THRESHOLD

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run person detection on a single frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            
        Returns:
            sv.Detections with only person detections
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[0],    # class 0 = person in COCO
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """Batch detection for multiple frames."""
        results_list = self.model(
            frames,
            conf=self.confidence,
            iou=self.iou,
            classes=[0],
            verbose=False,
        )
        return [sv.Detections.from_ultralytics(r) for r in results_list]
