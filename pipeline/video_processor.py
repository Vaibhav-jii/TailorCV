"""
StoreIQ Video Processor — Main orchestrator that runs the full pipeline.

Takes a video file → runs detection → tracking → zone classification →
event generation → outputs annotated video + events + heatmap.

This is the primary entry point for processing CCTV footage.
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, List
import supervision as sv
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.zone_manager import ZoneManager
from core.event_engine import EventEngine
from core.models import StoreEvent
from config.settings import settings

console = Console()


class VideoProcessor:
    """
    Main orchestrator that processes video files through the intelligence pipeline.
    
    Usage:
        processor = VideoProcessor(zones_config="config/zones.json")
        results = processor.process_video("video.mp4", output_path="annotated.mp4")
    """

    def __init__(
        self,
        zones_config: str = None,
        camera_id: str = "cam_01",
        store_id: str = "store_01",
        model_path: str = None,
    ):
        self.detector = PersonDetector(model_path=model_path)
        self.tracker = PersonTracker()
        self.zone_manager = ZoneManager(zones_config)
        self.event_engine = EventEngine(self.zone_manager)
        self.camera_id = camera_id
        self.store_id = store_id

        # Annotators for visualization (version-safe initialization)
        try:
            palette = sv.ColorPalette.from_hex(["#7c3aed", "#a855f7", "#c084fc", "#e9d5ff"])
            self.box_annotator = sv.BoxAnnotator(thickness=2, color=palette)
            self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=palette)
            self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60, color=palette)
        except (TypeError, AttributeError):
            # Fallback for different supervision versions
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)

        try:
            self.heat_annotator = sv.HeatMapAnnotator(
                position=sv.Position.BOTTOM_CENTER,
                opacity=0.5,
                radius=40,
            )
        except TypeError:
            self.heat_annotator = sv.HeatMapAnnotator()

        # Results storage
        self.all_events: List[StoreEvent] = []
        self._heatmap_sink: Optional[np.ndarray] = None

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        heatmap_output: Optional[str] = None,
        on_event: Optional[Callable[[StoreEvent], None]] = None,
        show_preview: bool = False,
        start_time: Optional[datetime] = None,
    ) -> dict:
        """
        Process a video file end-to-end.
        
        Args:
            video_path: Path to input CCTV video
            output_path: Path to save annotated output video (optional)
            heatmap_output: Path to save heatmap image (optional)
            on_event: Callback function for real-time event handling
            show_preview: Show live OpenCV preview window
            start_time: Simulated start timestamp (defaults to now)
            
        Returns:
            dict with processing results, metrics, events, and journeys
        """
        # ── Open video ──
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frame skip to achieve target processing FPS
        frame_skip = max(1, source_fps // settings.PROCESS_FPS)

        if start_time is None:
            start_time = datetime.now()

        # ── Video writer for annotated output ──
        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, settings.PROCESS_FPS, (width, height)
            )

        # ── Print header ──
        console.print()
        console.print("[bold magenta]╔══════════════════════════════════════════╗[/]")
        console.print("[bold magenta]║   StoreIQ — Video Processing Pipeline   ║[/]")
        console.print("[bold magenta]╚══════════════════════════════════════════╝[/]")
        console.print()
        console.print(f"  📹 Video:      [cyan]{Path(video_path).name}[/]")
        console.print(f"  📐 Resolution: {width}×{height}")
        console.print(f"  🎬 Source FPS:  {source_fps} → Processing at {settings.PROCESS_FPS} FPS (skip {frame_skip})")
        console.print(f"  📊 Total frames: {total_frames}")
        console.print(f"  🗺️  Zones:       {self.zone_manager.get_zone_count()} configured")
        console.print()

        # ── Processing loop ──
        frame_number = 0
        processed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing frames...", total=total_frames
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                progress.update(task, advance=1)

                # Skip frames for target FPS
                if frame_number % frame_skip != 0:
                    continue

                processed_count += 1

                # Calculate simulated timestamp
                elapsed_seconds = frame_number / source_fps
                timestamp = start_time + timedelta(seconds=elapsed_seconds)

                # ── Detection ──
                detections = self.detector.detect(frame)

                # ── Tracking ──
                detections = self.tracker.update(detections)

                # ── Event Generation ──
                events = self.event_engine.process_frame(
                    detections, frame_number, timestamp,
                    self.camera_id, self.store_id,
                )

                # Event callback
                if on_event:
                    for event in events:
                        on_event(event)

                # ── Annotate for output ──
                if writer or show_preview:
                    annotated = self._annotate_frame(frame, detections)
                    if writer:
                        writer.write(annotated)
                    if show_preview:
                        cv2.imshow("StoreIQ - Live Preview", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # ── Update heatmap ──
                if self._heatmap_sink is None:
                    self._heatmap_sink = np.zeros((height, width, 3), dtype=np.uint8)
                self._heatmap_sink = self.heat_annotator.annotate(
                    scene=self._heatmap_sink, detections=detections
                )

        # ── Finalize ──
        final_timestamp = start_time + timedelta(seconds=frame_number / source_fps)
        self.event_engine.finalize(frame_number, final_timestamp)

        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Save heatmap
        if heatmap_output and self._heatmap_sink is not None:
            Path(heatmap_output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(heatmap_output, self._heatmap_sink)
            console.print(f"  🔥 Heatmap saved to [cyan]{heatmap_output}[/]")

        # ── Build results ──
        metrics = self.event_engine.get_store_metrics()
        journeys = self.event_engine.get_all_journeys()
        self.all_events = self.event_engine.all_events

        results = {
            "video": str(video_path),
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "source_fps": source_fps,
            "process_fps": settings.PROCESS_FPS,
            "total_events": len(self.all_events),
            "metrics": metrics,
            "journeys": [j.model_dump() for j in journeys],
        }

        self._print_summary(metrics, journeys)
        return results

    def _annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Annotate a frame with zones, tracks, bounding boxes, and metrics overlay."""
        annotated = frame.copy()

        # Draw zones
        annotated = self.zone_manager.draw_zones(annotated)

        # Draw detections
        if detections.tracker_id is not None and len(detections) > 0:
            # Generate labels
            labels = []
            for i in range(len(detections)):
                tid = detections.tracker_id[i]
                conf = detections.confidence[i] if detections.confidence is not None else 0
                labels.append(f"#{tid} {conf:.0%}")

            annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(
                annotated, detections, labels=labels
            )

        # Metrics overlay (top-left)
        metrics = self.event_engine.get_store_metrics()
        overlay_lines = [
            f"Occupancy: {metrics['current_occupancy']}",
            f"Entries: {metrics['total_entries']}",
            f"Exits: {metrics['total_exits']}",
            f"Conversion: {metrics['conversion_rate']:.1%}",
            f"Billing: {metrics['reached_billing']}",
        ]

        # Dark background for overlay
        overlay_h = len(overlay_lines) * 28 + 16
        cv2.rectangle(annotated, (0, 0), (250, overlay_h), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (250, overlay_h), (124, 58, 237), 2)

        for i, text in enumerate(overlay_lines):
            cv2.putText(
                annotated, text, (10, 24 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return annotated

    def _print_summary(self, metrics: dict, journeys: List):
        """Print a rich summary table after processing."""
        console.print()
        console.print("[bold green]═══ Processing Complete ═══[/]")
        console.print()

        # Metrics table
        table = Table(title="Store Metrics", border_style="bright_magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white", justify="right")

        table.add_row("Total Entries", str(metrics["total_entries"]))
        table.add_row("Total Exits", str(metrics["total_exits"]))
        table.add_row("Current Occupancy", str(metrics["current_occupancy"]))
        table.add_row("Reached Billing", str(metrics["reached_billing"]))
        table.add_row("Purchased", str(metrics["purchased"]))
        table.add_row("Browsed Zones", str(metrics["browsed_zones"]))
        table.add_row("─" * 20, "─" * 10)
        table.add_row(
            "🎯 Conversion Rate",
            f"{metrics['conversion_rate']:.1%}",
        )
        table.add_row(
            "Billing Reach Rate",
            f"{metrics['billing_reach_rate']:.1%}",
        )
        table.add_row(
            "Avg Visit Duration",
            f"{metrics['avg_visit_duration']:.1f}s",
        )
        table.add_row("Total Events", str(len(self.event_engine.all_events)))

        console.print(table)

        # Zone occupancy
        zone_occ = self.event_engine.get_zone_occupancy()
        if zone_occ:
            console.print()
            ztable = Table(title="Zone Summary", border_style="bright_magenta")
            ztable.add_column("Zone", style="cyan")
            ztable.add_column("Current Occupancy", justify="right")
            for zone, count in zone_occ.items():
                ztable.add_row(zone, str(count))
            console.print(ztable)
