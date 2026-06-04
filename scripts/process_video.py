#!/usr/bin/env python3
"""
StoreIQ CLI — Process CCTV video files through the intelligence pipeline.

Usage:
    # Process a single video
    python scripts/process_video.py video data/videos/cam1.mp4 --zones config/zones.json

    # Process all videos in a directory
    python scripts/process_video.py batch data/videos/ --zones config/zones.json

    # Process with annotated output video
    python scripts/process_video.py video data/videos/cam1.mp4 -o data/output/annotated.mp4

    # Process with heatmap
    python scripts/process_video.py video data/videos/cam1.mp4 --heatmap data/output/heatmap.png
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from pipeline.video_processor import VideoProcessor
from storage.database import Database
from analytics.anomaly import AnomalyDetector
from config.settings import settings

app = typer.Typer(
    name="storeiq",
    help="StoreIQ — AI-Powered Store Intelligence from CCTV footage",
    add_completion=False,
)
console = Console()


@app.command()
def video(
    video_path: str = typer.Argument(..., help="Path to the CCTV video file"),
    zones: str = typer.Option(
        None, "--zones", "-z", help="Path to zones configuration JSON"
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Path to save annotated output video"
    ),
    heatmap: str = typer.Option(
        None, "--heatmap", help="Path to save heatmap image"
    ),
    camera_id: str = typer.Option("cam_01", "--camera", "-c", help="Camera ID"),
    store_id: str = typer.Option("store_01", "--store", "-s", help="Store ID"),
    save_json: str = typer.Option(
        None, "--save-json", help="Path to save results as JSON"
    ),
    preview: bool = typer.Option(False, "--preview", "-p", help="Show live preview"),
    db_path: str = typer.Option(None, "--db", help="Custom database path"),
    model: str = typer.Option(None, "--model", "-m", help="YOLOv8 model path"),
    detect_anomalies: bool = typer.Option(
        True, "--anomalies/--no-anomalies", help="Run anomaly detection after processing"
    ),
):
    """Process a single CCTV video file and extract store intelligence."""

    if not Path(video_path).exists():
        console.print(f"[red]✗ Error: Video file not found: {video_path}[/]")
        raise typer.Exit(1)

    zones_path = zones or str(settings.ZONES_CONFIG)
    if not Path(zones_path).exists():
        console.print(f"[yellow]⚠ Warning: Zones config not found: {zones_path}[/]")
        console.print("[yellow]  Processing without zone definitions. Use setup-zones command first.[/]")
        zones_path = None

    # Initialize processor
    processor = VideoProcessor(
        zones_config=zones_path,
        camera_id=camera_id,
        store_id=store_id,
        model_path=model,
    )

    # Process video
    results = processor.process_video(
        video_path=video_path,
        output_path=output,
        heatmap_output=heatmap,
        show_preview=preview,
        start_time=datetime.now(),
    )

    # Save to database
    db = Database(db_path)
    console.print("\n[cyan]Saving to database...[/]")

    db.insert_events_batch(processor.all_events)
    for journey in processor.event_engine.get_all_journeys():
        db.insert_journey(journey)
    db.record_processing_run(results)

    console.print(
        f"[green]✓ Saved {len(processor.all_events)} events and "
        f"{len(processor.event_engine.get_all_journeys())} journeys[/]"
    )

    # Run anomaly detection
    if detect_anomalies:
        console.print("\n[cyan]Running anomaly detection...[/]")
        detector = AnomalyDetector(db)
        detector.compute_baselines()
        anomalies = detector.run_detection(results.get("metrics"))
        if anomalies:
            console.print(f"[yellow]⚠ Found {len(anomalies)} anomalies:[/]")
            for alert in anomalies:
                color = {"low": "green", "medium": "yellow", "high": "red"}[alert.severity]
                console.print(f"  [{color}]• [{alert.severity.upper()}] {alert.message}[/]")
        else:
            console.print("[green]✓ No anomalies detected[/]")

    # Save JSON results if requested
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]✓ Results saved to {save_json}[/]")

    # Next steps
    console.print("\n[bold magenta]═══ Next Steps ═══[/]")
    console.print("  📊 Dashboard:  [cyan]streamlit run dashboard/app.py[/]")
    console.print("  🔌 API:        [cyan]uvicorn api.app:app --reload --port 8000[/]")
    console.print("  📖 API Docs:   [cyan]http://localhost:8000/docs[/]")


@app.command()
def batch(
    videos_dir: str = typer.Argument(..., help="Directory containing video files"),
    zones: str = typer.Option(None, "--zones", "-z", help="Zones config path"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    db_path: str = typer.Option(None, "--db", help="Custom database path"),
    model: str = typer.Option(None, "--model", "-m", help="YOLOv8 model path"),
):
    """Process all video files in a directory."""

    video_dir = Path(videos_dir)
    if not video_dir.exists():
        console.print(f"[red]✗ Error: Directory not found: {videos_dir}[/]")
        raise typer.Exit(1)

    # Find video files
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")
    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.glob(ext))
    video_files.sort()

    if not video_files:
        console.print(f"[yellow]No video files found in {videos_dir}[/]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Found {len(video_files)} video files:[/]")
    for vf in video_files:
        console.print(f"  📹 {vf.name}")

    zones_path = zones or str(settings.ZONES_CONFIG)
    if not Path(zones_path).exists():
        zones_path = None

    db = Database(db_path)

    for i, video_file in enumerate(video_files, 1):
        console.print(
            f"\n[bold]{'═' * 50}[/]"
            f"\n[bold]  Video {i}/{len(video_files)}: {video_file.name}[/]"
            f"\n[bold]{'═' * 50}[/]"
        )

        out_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = str(Path(output_dir) / f"annotated_{video_file.stem}.mp4")

        heatmap_path = None
        if output_dir:
            heatmap_path = str(Path(output_dir) / f"heatmap_{video_file.stem}.png")

        processor = VideoProcessor(
            zones_config=zones_path,
            camera_id=f"cam_{i:02d}",
            model_path=model,
        )

        results = processor.process_video(
            video_path=str(video_file),
            output_path=out_path,
            heatmap_output=heatmap_path,
            start_time=datetime.now(),
        )

        # Save results
        db.insert_events_batch(processor.all_events)
        for journey in processor.event_engine.get_all_journeys():
            db.insert_journey(journey)
        db.record_processing_run(results)

    # Final summary
    console.print(f"\n[bold green]{'═' * 50}[/]")
    console.print(f"[bold green]  All {len(video_files)} videos processed![/]")
    console.print(f"[bold green]{'═' * 50}[/]")

    stats = db.get_stats()
    console.print(f"\n  Events:    {stats['total_events']}")
    console.print(f"  Journeys:  {stats['total_journeys']}")

    # Run anomaly detection on aggregate data
    detector = AnomalyDetector(db)
    detector.compute_baselines()
    anomalies = detector.run_detection()
    if anomalies:
        console.print(f"\n[yellow]⚠ {len(anomalies)} anomalies detected[/]")

    console.print("\n[bold magenta]═══ Next Steps ═══[/]")
    console.print("  📊 Dashboard:  [cyan]streamlit run dashboard/app.py[/]")
    console.print("  🔌 API:        [cyan]uvicorn api.app:app --reload --port 8000[/]")


@app.command(name="setup-zones")
def setup_zones(
    video_path: str = typer.Argument(..., help="Path to video file for zone setup"),
    output: str = typer.Option(
        None, "--output", "-o", help="Output path for zones config"
    ),
):
    """
    Interactive zone setup tool.
    
    Opens a video frame and lets you draw polygon zones by clicking.
    
    Controls:
    - Click to add polygon points
    - 'n' = new zone (prompts for name in terminal)
    - 'f' = finish current zone
    - 's' = save configuration
    - 'u' = undo last point
    - 'q' = quit
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from zone_setup_interactive import ZoneSetup

    out_path = output or str(settings.ZONES_CONFIG)
    setup = ZoneSetup(video_path, out_path)
    setup.run()


if __name__ == "__main__":
    app()
