"""
StoreIQ Zone Manager — Manages store zone polygons and line crossings.

Handles:
- Loading/saving zone configurations from JSON
- Point-in-polygon checks for zone membership
- Line crossing detection for entry/exit
- Interactive zone setup support
"""

import json
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ZoneManager:
    """
    Manages polygon zones and entry/exit lines for a camera view.
    
    Each zone is a polygon defined by corner points. Persons are classified
    into zones based on their bottom-center point (foot position).
    """

    def __init__(self, zones_config_path: str = None):
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.zone_polygons: Dict[str, np.ndarray] = {}
        self.zone_types: Dict[str, str] = {}
        self.frame_resolution: Tuple[int, int] = (640, 480)

        if zones_config_path and Path(zones_config_path).exists():
            self.load_zones(zones_config_path)

    def load_zones(self, config_path: str):
        """Load zone definitions from a JSON configuration file."""
        with open(config_path, "r") as f:
            config = json.load(f)

        for zone_config in config.get("zones", []):
            name = zone_config["name"]
            polygon = np.array(zone_config["polygon"], dtype=np.int32)
            zone_type = zone_config.get("type", "general")

            self.zone_polygons[name] = polygon
            self.zone_types[name] = zone_type
            self.zones[name] = sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=[sv.Position.BOTTOM_CENTER],
            )

    def check_zones(self, detections: sv.Detections) -> Dict[str, np.ndarray]:
        """
        Check which zones each detected person is in.
        
        Returns:
            Dict mapping zone_name → boolean mask (True for each detection in that zone)
        """
        results = {}
        for name, zone in self.zones.items():
            mask = zone.trigger(detections)
            results[name] = mask
        return results

    def get_person_zone(
        self, detections: sv.Detections, person_idx: int
    ) -> Optional[str]:
        """Get the zone name for a specific person by index."""
        zone_results = self.check_zones(detections)
        for zone_name, mask in zone_results.items():
            if mask[person_idx]:
                return zone_name
        return None

    def get_zone_type(self, zone_name: str) -> str:
        """Get the type of a zone (e.g., 'billing', 'skincare')."""
        return self.zone_types.get(zone_name, "general")

    def is_billing_zone(self, zone_name: str) -> bool:
        """Check if a zone is the billing/checkout area."""
        return self.zone_types.get(zone_name, "") == "billing"

    def get_zone_names(self) -> List[str]:
        """Get all configured zone names."""
        return list(self.zones.keys())

    def get_zone_count(self) -> int:
        """Get number of configured zones."""
        return len(self.zones)

    def save_zones(self, config_path: str):
        """Save current zone configuration to JSON."""
        config = {
            "zones": [
                {
                    "name": name,
                    "polygon": self.zone_polygons[name].tolist(),
                    "type": self.zone_types.get(name, "general"),
                }
                for name in self.zone_polygons
            ]
        }

        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on a frame for visualization."""
        import cv2

        annotated = frame.copy()

        # Color map by zone type
        color_map = {
            "entrance": (0, 255, 0),      # green
            "exit": (0, 0, 255),           # red
            "billing": (0, 165, 255),      # orange
            "skincare": (237, 58, 124),    # pink
            "makeup": (237, 149, 58),      # amber
            "fragrance": (58, 237, 186),   # teal
            "haircare": (186, 58, 237),    # purple
            "general": (200, 200, 200),    # gray
        }

        for name, polygon in self.zone_polygons.items():
            zone_type = self.zone_types.get(name, "general")
            color = color_map.get(zone_type, (200, 200, 200))

            # Draw filled polygon with transparency
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

            # Draw border
            cv2.polylines(annotated, [polygon], True, color, 2)

            # Label
            centroid = polygon.mean(axis=0).astype(int)
            label = f"{name} ({zone_type})"
            cv2.putText(
                annotated, label, tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )

        return annotated
