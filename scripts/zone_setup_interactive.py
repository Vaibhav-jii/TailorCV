#!/usr/bin/env python3
"""
StoreIQ Interactive Zone Setup — Draw polygon zones on video frames.

Usage:
    python scripts/zone_setup_interactive.py <video_path> [output_path]

Controls:
    Click       → Add point to current polygon
    'n'         → Start a new zone (enter name in terminal)
    'f'         → Finish current zone (close polygon)
    'u'         → Undo last point
    'r'         → Reset all zones
    's'         → Save zones to config file
    'q'         → Quit

Each zone needs:
    - A name (e.g., "skincare", "billing", "entrance")
    - A type (e.g., "skincare", "billing", "entrance", "general")
    - At least 3 polygon points
"""

import cv2
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


# Zone type colors (BGR)
ZONE_COLORS = {
    "entrance": (0, 255, 0),
    "exit": (0, 0, 255),
    "billing": (0, 165, 255),
    "skincare": (237, 58, 124),
    "makeup": (237, 149, 58),
    "fragrance": (58, 237, 186),
    "haircare": (186, 58, 237),
    "general": (200, 200, 200),
}


class ZoneSetup:
    """Interactive tool to define store zones on a camera frame."""

    def __init__(self, video_path: str, output_path: str = None):
        self.video_path = video_path
        self.output_path = output_path or str(settings.ZONES_CONFIG)
        self.frame = self._get_first_frame()
        self.zones: Dict[str, dict] = {}
        self.current_zone_name: str = ""
        self.current_zone_type: str = "general"
        self.current_points: List[Tuple[int, int]] = []
        self.window_name = "StoreIQ Zone Setup"

    def _get_first_frame(self) -> np.ndarray:
        """Extract the first frame from the video."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read video: {self.video_path}")
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to add polygon points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self._redraw()

    def _redraw(self):
        """Redraw the frame with all zones and current points."""
        display = self.frame.copy()

        # Draw completed zones
        for name, zone_data in self.zones.items():
            pts = np.array(zone_data["points"], dtype=np.int32)
            zone_type = zone_data.get("type", "general")
            color = ZONE_COLORS.get(zone_type, (200, 200, 200))

            # Filled polygon with transparency
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

            # Border
            cv2.polylines(display, [pts], True, color, 2)

            # Label
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(
                display, f"{name} ({zone_type})", tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # Draw current points
        for i, pt in enumerate(self.current_points):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.circle(display, pt, 8, (255, 255, 255), 1)
            if i > 0:
                cv2.line(display, self.current_points[i - 1], pt, (0, 255, 0), 2)

        # Close polygon preview
        if len(self.current_points) > 2:
            cv2.line(
                display, self.current_points[-1], self.current_points[0],
                (0, 255, 0), 1, cv2.LINE_AA,
            )

        # Instructions panel (top)
        panel_h = 80
        cv2.rectangle(display, (0, 0), (display.shape[1], panel_h), (0, 0, 0), -1)

        instructions = [
            f"Zone: {self.current_zone_name or '(none)'} | Type: {self.current_zone_type} | Points: {len(self.current_points)}",
            "Click=add point | n=new zone | f=finish zone | u=undo | s=save | q=quit",
            f"Completed zones: {', '.join(self.zones.keys()) if self.zones else 'none'}",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                display, text, (10, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

        cv2.imshow(self.window_name, display)

    def run(self):
        """Run the interactive zone setup."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()

        print("\n╔══════════════════════════════════════╗")
        print("║   StoreIQ — Interactive Zone Setup   ║")
        print("╚══════════════════════════════════════╝")
        print("\nControls:")
        print("  Click  → Add polygon point")
        print("  'n'    → New zone (enter name in terminal)")
        print("  'f'    → Finish current zone")
        print("  'u'    → Undo last point")
        print("  'r'    → Reset all zones")
        print("  's'    → Save to config file")
        print("  'q'    → Quit\n")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord("q"):
                if self.zones:
                    save = input("\nSave zones before quitting? (y/n): ").strip().lower()
                    if save == "y":
                        self._save()
                break

            elif key == ord("n"):
                # New zone
                name = input("\nEnter zone name: ").strip()
                if not name:
                    print("Zone name cannot be empty.")
                    continue

                print("Zone types: entrance, exit, skincare, makeup, fragrance, haircare, billing, general")
                zone_type = input("Enter zone type: ").strip() or "general"

                self.current_zone_name = name
                self.current_zone_type = zone_type
                self.current_points = []
                print(f"Drawing zone: '{name}' (type: {zone_type})")
                print("Click on the image to add polygon points, then press 'f' to finish.")
                self._redraw()

            elif key == ord("f"):
                # Finish current zone
                if len(self.current_points) < 3:
                    print("Need at least 3 points to create a zone.")
                    continue
                if not self.current_zone_name:
                    print("No zone started. Press 'n' to create a new zone first.")
                    continue

                self.zones[self.current_zone_name] = {
                    "points": list(self.current_points),
                    "type": self.current_zone_type,
                }
                print(f"✓ Zone '{self.current_zone_name}' saved with {len(self.current_points)} points")
                self.current_points = []
                self.current_zone_name = ""
                self._redraw()

            elif key == ord("u"):
                # Undo last point
                if self.current_points:
                    self.current_points.pop()
                    self._redraw()

            elif key == ord("r"):
                # Reset all
                confirm = input("\nReset all zones? (y/n): ").strip().lower()
                if confirm == "y":
                    self.zones = {}
                    self.current_points = []
                    self.current_zone_name = ""
                    self._redraw()
                    print("All zones reset.")

            elif key == ord("s"):
                self._save()

        cv2.destroyAllWindows()

    def _save(self):
        """Save zones to config file."""
        config = {
            "zones": [
                {
                    "name": name,
                    "polygon": data["points"],
                    "type": data["type"],
                }
                for name, data in self.zones.items()
            ]
        }

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Zones saved to {self.output_path}")
        print(f"  Zones defined: {list(self.zones.keys())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zone_setup_interactive.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    setup = ZoneSetup(video_path, output_path)
    setup.run()
