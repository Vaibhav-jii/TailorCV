#!/usr/bin/env python3
"""
StoreIQ Live Feed — Real-time CCTV Intelligence Viewer.

Plays video with live person detection, tracking, and a premium HUD showing:
  • Big live people counter (currently in store)
  • Total entries / exits
  • Zone occupancy breakdown
  • Scrolling event ticker (last 8 events)
  • Conversion rate
  • Per-person bounding boxes with track IDs + zone labels

Usage:
    python scripts/live_feed.py "data/videos/CAM 1.mp4"
    python scripts/live_feed.py "data/videos/CAM 1.mp4" --zones config/zones.json
    python scripts/live_feed.py "data/videos/CAM 1.mp4" --fullscreen
    python scripts/live_feed.py "data/videos/CAM 1.mp4" --speed 2.0

Controls:
    SPACE   Pause / Resume
    Q / ESC Quit
    + / -   Speed up / slow down
    F       Toggle fullscreen
    R       Reset counters
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from typing import Optional
import supervision as sv

from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.zone_manager import ZoneManager
from core.event_engine import EventEngine
from core.models import StoreEvent
from config.settings import settings


# ═══════════════════════════════════════════════════════════
# Color palette & styling constants
# ═══════════════════════════════════════════════════════════

# BGR colors
PURPLE_DARK    = (117, 37, 78)    # #4E2575 — deep purple
PURPLE_MAIN    = (237, 58, 124)   # #7C3AED
PURPLE_LIGHT   = (252, 132, 192)  # #C084FC
PURPLE_GLOW    = (250, 165, 168)  # #A8A5FA
GREEN_ACCENT   = (100, 220, 60)   # #3CDC64
RED_ACCENT     = (80, 80, 240)    # #F05050
ORANGE_ACCENT  = (60, 165, 255)   # #FFA53C
WHITE          = (255, 255, 255)
LIGHT_GRAY     = (200, 210, 225)  # #E1D2C8
MID_GRAY       = (140, 150, 170)  # #AA9690
DARK_BG        = (25, 22, 18)     # #121619
PANEL_BG       = (40, 35, 30)     # #1E2328
PANEL_BORDER   = (80, 65, 55)     # #374150

# Event type → color + emoji
EVENT_STYLE = {
    "ENTRY":                 (GREEN_ACCENT,  "→ IN"),
    "EXIT":                  (RED_ACCENT,    "← OUT"),
    "ZONE_ENTER":            (PURPLE_LIGHT,  "▶ ZONE"),
    "ZONE_EXIT":             (MID_GRAY,      "◀ ZONE"),
    "ZONE_DWELL":            (ORANGE_ACCENT, "⏱ DWELL"),
    "BILLING_QUEUE_JOIN":    (ORANGE_ACCENT, "🛒 QUEUE"),
    "BILLING_QUEUE_ABANDON": (RED_ACCENT,    "✗ ABANDON"),
    "PURCHASE_INFERRED":     (GREEN_ACCENT,  "💰 PURCHASE"),
    "REENTRY":               (PURPLE_GLOW,   "↩ REENTRY"),
}


class LiveFeedHUD:
    """Draws a premium heads-up display over the video frame."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.event_log: deque = deque(maxlen=8)
        self.start_time = time.time()

    def add_event(self, event: StoreEvent):
        """Add an event to the scrolling ticker."""
        self.event_log.appendleft(event)

    def draw(
        self,
        frame: np.ndarray,
        metrics: dict,
        zone_occupancy: dict,
        detections: sv.Detections,
        paused: bool = False,
        speed: float = 1.0,
        frame_number: int = 0,
        total_frames: int = 0,
    ) -> np.ndarray:
        """Draw the full HUD overlay on the frame."""
        out = frame.copy()

        # ── Left panel: Big counter + KPIs ──
        self._draw_left_panel(out, metrics)

        # ── Right panel: Zone occupancy ──
        self._draw_right_panel(out, zone_occupancy)

        # ── Bottom bar: Event ticker ──
        self._draw_event_ticker(out)

        # ── Top bar: Status strip ──
        self._draw_top_bar(out, paused, speed, frame_number, total_frames)

        # ── Bottom-right: Progress ──
        self._draw_progress(out, frame_number, total_frames)

        return out

    def _draw_panel_bg(self, frame, x, y, w, h, alpha=0.82):
        """Draw a dark semi-transparent panel with a border."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), DARK_BG, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL_BORDER, 1)

    def _draw_left_panel(self, frame, metrics):
        """Big occupancy counter + KPIs on the left side."""
        pw, ph = 260, 330
        px, py = 12, 60
        self._draw_panel_bg(frame, px, py, pw, ph, alpha=0.85)

        # Title
        cv2.putText(frame, "LIVE INTELLIGENCE", (px + 12, py + 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, PURPLE_LIGHT, 1, cv2.LINE_AA)

        # ── BIG counter: current occupancy ──
        occ = metrics.get("current_occupancy", 0)
        occ_str = str(occ)

        # Big number
        font_scale = 3.2 if occ < 100 else 2.4
        (tw, th), _ = cv2.getTextSize(occ_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
        cx = px + pw // 2 - tw // 2
        cy = py + 105

        # Glow effect
        for offset in range(3, 0, -1):
            glow_color = (
                min(255, PURPLE_MAIN[0] + offset * 30),
                min(255, PURPLE_MAIN[1] + offset * 15),
                min(255, PURPLE_MAIN[2] + offset * 15),
            )
            cv2.putText(frame, occ_str, (cx, cy),
                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow_color,
                         4 + offset, cv2.LINE_AA)

        cv2.putText(frame, occ_str, (cx, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 4, cv2.LINE_AA)

        # Label under the number
        label = "PEOPLE IN STORE"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, label, (px + pw // 2 - lw // 2, cy + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, MID_GRAY, 1, cv2.LINE_AA)

        # ── KPI grid ──
        kpis = [
            ("ENTERED", str(metrics.get("total_entries", 0)), GREEN_ACCENT),
            ("EXITED",  str(metrics.get("total_exits", 0)),   RED_ACCENT),
            ("BILLING", str(metrics.get("reached_billing", 0)), ORANGE_ACCENT),
            ("BOUGHT",  str(metrics.get("purchased", 0)),     GREEN_ACCENT),
        ]

        kpi_start_y = cy + 55
        col_w = pw // 2
        for i, (label, value, color) in enumerate(kpis):
            col = i % 2
            row = i // 2
            kx = px + 16 + col * col_w
            ky = kpi_start_y + row * 58

            # Value
            cv2.putText(frame, value, (kx, ky),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            # Label
            cv2.putText(frame, label, (kx, ky + 18),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, MID_GRAY, 1, cv2.LINE_AA)

        # ── Conversion rate bar ──
        conv = metrics.get("conversion_rate", 0)
        bar_y = kpi_start_y + 130
        cv2.putText(frame, f"CONVERSION  {conv:.1%}", (px + 16, bar_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, LIGHT_GRAY, 1, cv2.LINE_AA)

        bar_x = px + 16
        bar_w = pw - 32
        bar_h = 8
        bar_y2 = bar_y + 8
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h),
                       PANEL_BORDER, -1)
        fill_w = int(bar_w * min(conv, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill_w, bar_y2 + bar_h),
                           GREEN_ACCENT, -1)

    def _draw_right_panel(self, frame, zone_occupancy):
        """Zone occupancy breakdown on the right side."""
        if not zone_occupancy:
            return

        pw = 200
        line_h = 36
        ph = 40 + len(zone_occupancy) * line_h + 10
        px = self.w - pw - 12
        py = 60
        self._draw_panel_bg(frame, px, py, pw, ph, alpha=0.82)

        cv2.putText(frame, "ZONE OCCUPANCY", (px + 12, py + 26),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE_LIGHT, 1, cv2.LINE_AA)

        # Zone color map
        zone_colors = {
            "entrance":  GREEN_ACCENT,
            "billing":   ORANGE_ACCENT,
            "skincare":  (237, 58, 124),   # pink
            "makeup":    (58, 149, 237),    # amber-ish in BGR
            "fragrance": (186, 237, 58),    # teal
            "haircare":  (237, 58, 186),    # purple
        }

        for i, (zone_name, count) in enumerate(zone_occupancy.items()):
            zy = py + 45 + i * line_h
            color = zone_colors.get(zone_name, LIGHT_GRAY)

            # Zone indicator dot
            cv2.circle(frame, (px + 18, zy - 4), 5, color, -1, cv2.LINE_AA)

            # Zone name
            cv2.putText(frame, zone_name.upper(), (px + 30, zy),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, LIGHT_GRAY, 1, cv2.LINE_AA)

            # Count (right-aligned)
            count_str = str(count)
            (cw, _), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            count_color = WHITE if count > 0 else MID_GRAY
            cv2.putText(frame, count_str, (px + pw - cw - 16, zy + 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, count_color, 2, cv2.LINE_AA)

    def _draw_event_ticker(self, frame):
        """Scrolling event log at the bottom."""
        if not self.event_log:
            return

        ticker_h = min(len(self.event_log) * 24 + 30, 230)
        ticker_y = self.h - ticker_h - 8
        ticker_w = self.w - 24
        self._draw_panel_bg(frame, 12, ticker_y, ticker_w, ticker_h, alpha=0.80)

        cv2.putText(frame, "LIVE EVENTS", (24, ticker_y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, PURPLE_LIGHT, 1, cv2.LINE_AA)

        for i, event in enumerate(self.event_log):
            if i >= 8:
                break
            ey = ticker_y + 40 + i * 22
            if ey + 10 > self.h - 8:
                break

            etype = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            style = EVENT_STYLE.get(etype, (LIGHT_GRAY, etype))
            color, prefix = style

            # Fade older events
            age_factor = max(0.4, 1.0 - i * 0.08)
            faded = tuple(int(c * age_factor) for c in color)

            ts = event.timestamp.strftime("%H:%M:%S") if event.timestamp else ""
            zone_str = f" [{event.zone}]" if event.zone else ""
            line = f"{ts}  {prefix}  {event.person_id}{zone_str}"

            cv2.putText(frame, line, (24, ey),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, faded, 1, cv2.LINE_AA)

    def _draw_top_bar(self, frame, paused, speed, frame_number, total_frames):
        """Status strip at the top."""
        bar_h = 44
        self._draw_panel_bg(frame, 0, 0, self.w, bar_h, alpha=0.85)

        # Left: StoreIQ brand
        cv2.putText(frame, "StoreIQ", (14, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.75, PURPLE_LIGHT, 2, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (120, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN_ACCENT, 2, cv2.LINE_AA)

        # Blinking dot for LIVE
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (108, 25), 5, GREEN_ACCENT, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (108, 25), 5, (0, 80, 0), -1, cv2.LINE_AA)

        # Center: Paused indicator
        if paused:
            cv2.putText(frame, "|| PAUSED", (self.w // 2 - 50, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE_ACCENT, 2, cv2.LINE_AA)

        # Right: Speed + frame info
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        info = f"{speed:.1f}x | Frame {frame_number}/{total_frames} | {elapsed_str}"
        (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, info, (self.w - iw - 14, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, MID_GRAY, 1, cv2.LINE_AA)

    def _draw_progress(self, frame, frame_number, total_frames):
        """Thin progress bar at the very bottom."""
        if total_frames <= 0:
            return
        bar_h = 4
        bar_y = self.h - bar_h
        progress = frame_number / total_frames
        fill_w = int(self.w * progress)

        cv2.rectangle(frame, (0, bar_y), (self.w, self.h), DARK_BG, -1)
        if fill_w > 0:
            cv2.rectangle(frame, (0, bar_y), (fill_w, self.h), PURPLE_MAIN, -1)


# ═══════════════════════════════════════════════════════════
# Enhanced annotator for live feed
# ═══════════════════════════════════════════════════════════

def annotate_detections(frame, detections, zone_manager, event_engine):
    """Draw bounding boxes, track IDs, zones, and motion traces."""
    if detections.tracker_id is None or len(detections) == 0:
        return frame

    for i in range(len(detections)):
        tid = int(detections.tracker_id[i])
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

        # Determine color based on zone
        person_state = event_engine.active_tracks.get(tid)
        current_zone = person_state.current_zone if person_state else None

        if current_zone:
            zone_type = zone_manager.get_zone_type(current_zone)
            if zone_type == "billing":
                box_color = ORANGE_ACCENT
            elif zone_type == "entrance":
                box_color = GREEN_ACCENT
            else:
                box_color = PURPLE_LIGHT
        else:
            box_color = PURPLE_MAIN

        # Draw bounding box with rounded corners effect
        thickness = 2
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, thickness, cv2.LINE_AA)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, thickness, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, thickness, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, thickness, cv2.LINE_AA)

        # Label background
        zone_label = f" [{current_zone}]" if current_zone else ""
        label = f"#{tid}{zone_label}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

        label_bg_y1 = max(0, y1 - lh - 10)
        label_bg_y2 = y1 - 2
        cv2.rectangle(frame, (x1, label_bg_y1), (x1 + lw + 8, label_bg_y2), DARK_BG, -1)
        cv2.rectangle(frame, (x1, label_bg_y1), (x1 + lw + 8, label_bg_y2), box_color, 1)
        cv2.putText(frame, label, (x1 + 4, label_bg_y2 - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

        # Center dot at foot position
        foot_x = (x1 + x2) // 2
        foot_y = y2
        cv2.circle(frame, (foot_x, foot_y), 3, box_color, -1, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════

app = typer.Typer(add_completion=False)


@app.command()
def main(
    video_path: str = typer.Argument(..., help="Path to the CCTV video file"),
    zones: str = typer.Option(None, "--zones", "-z", help="Path to zones JSON config"),
    camera_id: str = typer.Option("cam_01", "--camera", "-c", help="Camera ID"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier"),
    fullscreen: bool = typer.Option(False, "--fullscreen", "-f", help="Launch in fullscreen"),
    model: str = typer.Option(None, "--model", "-m", help="Custom YOLO model path"),
    scale: float = typer.Option(1.0, "--scale", help="Display scale (0.5=half, 2.0=double)"),
):
    """
    🎬 StoreIQ Live Feed — Watch CCTV with real-time people counting & analytics.
    
    Shows live detection, tracking, zone occupancy, and a scrolling event ticker
    as the video plays. Press SPACE to pause, Q to quit.
    """
    from rich.console import Console
    console = Console()

    if not Path(video_path).exists():
        console.print(f"[red]✗ Error: Video not found: {video_path}[/]")
        raise typer.Exit(1)

    zones_path = zones or str(settings.ZONES_CONFIG)
    if not Path(zones_path).exists():
        console.print(f"[yellow]⚠ No zones config at {zones_path}. Running without zones.[/]")
        zones_path = None

    # Initialize pipeline
    console.print()
    console.print("[bold magenta]╔══════════════════════════════════════════╗[/]")
    console.print("[bold magenta]║     StoreIQ — Live Intelligence Feed    ║[/]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/]")
    console.print()

    console.print("  [cyan]Loading YOLO model...[/]", end=" ")
    detector = PersonDetector(model_path=model)
    console.print("[green]✓[/]")

    console.print("  [cyan]Initializing tracker...[/]", end=" ")
    tracker = PersonTracker()
    console.print("[green]✓[/]")

    console.print("  [cyan]Loading zones...[/]", end=" ")
    zone_manager = ZoneManager(zones_path)
    console.print(f"[green]✓ ({zone_manager.get_zone_count()} zones)[/]")

    event_engine = EventEngine(zone_manager)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"[red]✗ Cannot open video: {video_path}[/]")
        raise typer.Exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, source_fps // settings.PROCESS_FPS)

    console.print(f"\n  📹 [cyan]{Path(video_path).name}[/]")
    console.print(f"  📐 {width}×{height} @ {source_fps} FPS")
    console.print(f"  🔍 Processing every {frame_skip}th frame ({settings.PROCESS_FPS} FPS)")
    console.print(f"  🗺️  {zone_manager.get_zone_count()} zones loaded")

    # Display sizing
    disp_w = int(width * scale)
    disp_h = int(height * scale)

    console.print(f"\n  [bold green]Starting live feed... Press Q or ESC to quit.[/]")
    console.print(f"  [dim]Controls: SPACE=pause, +/-=speed, F=fullscreen[/]\n")

    # HUD
    hud = LiveFeedHUD(width, height)

    # Window setup
    window_name = "StoreIQ - Live Intelligence Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # State
    paused = False
    current_speed = speed
    frame_number = 0
    start_time = datetime.now()
    target_delay = 1.0 / source_fps  # base delay per frame

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # End of video — show final frame for a bit
                    console.print("\n[bold green]═══ Video Complete ═══[/]")
                    metrics = event_engine.get_store_metrics()
                    console.print(f"  Total Entries:    {metrics['total_entries']}")
                    console.print(f"  Purchased:        {metrics['purchased']}")
                    console.print(f"  Conversion Rate:  {metrics['conversion_rate']:.1%}")
                    console.print(f"\n  Press any key in the video window to close...")
                    cv2.waitKey(0)
                    break

                frame_number += 1

                # Process detection on selected frames
                if frame_number % frame_skip == 0:
                    elapsed_seconds = frame_number / source_fps
                    timestamp = start_time + timedelta(seconds=elapsed_seconds)

                    # Detect
                    detections = detector.detect(frame)

                    # Track
                    detections = tracker.update(detections)

                    # Generate events
                    events = event_engine.process_frame(
                        detections, frame_number, timestamp,
                        camera_id, "store_01",
                    )

                    # Add new events to HUD ticker
                    for event in events:
                        hud.add_event(event)
                        # Print notable events to terminal too
                        etype = event.event_type.value
                        if etype in ("ENTRY", "EXIT", "PURCHASE_INFERRED", "BILLING_QUEUE_JOIN"):
                            style = EVENT_STYLE.get(etype, (None, etype))
                            console.print(
                                f"  [dim]{timestamp.strftime('%H:%M:%S')}[/] "
                                f"[bold]{style[1]}[/]  {event.person_id}"
                                f"{'  [' + event.zone + ']' if event.zone else ''}"
                            )

                    # Draw zones
                    annotated = zone_manager.draw_zones(frame)

                    # Draw detections with our custom style
                    annotated = annotate_detections(
                        annotated, detections, zone_manager, event_engine
                    )

                    # Draw HUD
                    metrics = event_engine.get_store_metrics()
                    zone_occ = event_engine.get_zone_occupancy()

                    annotated = hud.draw(
                        annotated, metrics, zone_occ, detections,
                        paused=paused, speed=current_speed,
                        frame_number=frame_number, total_frames=total_frames,
                    )

                    last_annotated = annotated
                else:
                    # For skipped frames, just show the raw frame with zones + HUD
                    # but keep last detection overlay
                    if 'last_annotated' not in dir() or last_annotated is None:
                        annotated = zone_manager.draw_zones(frame)
                        metrics = event_engine.get_store_metrics()
                        zone_occ = event_engine.get_zone_occupancy()
                        annotated = hud.draw(
                            annotated, metrics, zone_occ,
                            sv.Detections.empty(),
                            paused=paused, speed=current_speed,
                            frame_number=frame_number, total_frames=total_frames,
                        )
                    else:
                        annotated = last_annotated

            else:
                # Paused: keep showing last frame
                if 'last_annotated' in dir() and last_annotated is not None:
                    annotated = last_annotated.copy()
                    # Redraw HUD with paused state
                    metrics = event_engine.get_store_metrics()
                    zone_occ = event_engine.get_zone_occupancy()
                    base = frame.copy() if 'frame' in dir() else last_annotated
                    annotated = zone_manager.draw_zones(base)
                    annotated = annotate_detections(
                        annotated,
                        detections if 'detections' in dir() else sv.Detections.empty(),
                        zone_manager, event_engine
                    )
                    annotated = hud.draw(
                        annotated, metrics, zone_occ,
                        detections if 'detections' in dir() else sv.Detections.empty(),
                        paused=True, speed=current_speed,
                        frame_number=frame_number, total_frames=total_frames,
                    )
                else:
                    annotated = np.zeros((height, width, 3), dtype=np.uint8)

            # Display
            if scale != 1.0:
                display = cv2.resize(annotated, (disp_w, disp_h))
            else:
                display = annotated

            cv2.imshow(window_name, display)

            # Delay (adjusted for speed)
            delay_ms = max(1, int(target_delay * 1000 / current_speed))
            key = cv2.waitKey(delay_ms) & 0xFF

            # ── Key handling ──
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE = pause
                paused = not paused
                if paused:
                    console.print("  [yellow]⏸  Paused[/]")
                else:
                    console.print("  [green]▶  Resumed[/]")
            elif key == ord('+') or key == ord('='):
                current_speed = min(8.0, current_speed * 1.5)
                console.print(f"  [cyan]⏩ Speed: {current_speed:.1f}x[/]")
            elif key == ord('-') or key == ord('_'):
                current_speed = max(0.1, current_speed / 1.5)
                console.print(f"  [cyan]⏪ Speed: {current_speed:.1f}x[/]")
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, disp_w, disp_h)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted by user.[/]")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Final summary
        event_engine.finalize(frame_number, start_time + timedelta(seconds=frame_number / source_fps))
        metrics = event_engine.get_store_metrics()

        console.print()
        console.print("[bold magenta]═══ Session Summary ═══[/]")
        console.print(f"  👥 Total Entries:    {metrics['total_entries']}")
        console.print(f"  🚶 Total Exits:      {metrics['total_exits']}")
        console.print(f"  🛒 Reached Billing:  {metrics['reached_billing']}")
        console.print(f"  💰 Purchased:        {metrics['purchased']}")
        console.print(f"  🎯 Conversion Rate:  {metrics['conversion_rate']:.1%}")
        console.print(f"  📊 Total Events:     {len(event_engine.all_events)}")
        console.print()


if __name__ == "__main__":
    app()
