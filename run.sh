#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Store Intelligence — Pipeline Runner
# 
# One command to process all clips → events
# Usage: ./run.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════╗"
echo "║  Store Intelligence — Pipeline Runner    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Detect python command
PYTHON=python3
if ! command -v python3 &> /dev/null; then
    PYTHON=python
fi

# Step 1: Load POS transactions
echo ""
echo "═══ Step 1: Loading POS Transactions ═══"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from storage.database import Database
db = Database()
db.clear_all()
import glob
for csv_file in glob.glob('data/new_data/**/POS*.csv', recursive=True):
    n = db.load_pos_csv(csv_file, store_id='STORE_001')
    print(f'  ✓ Loaded {n} POS transactions from {csv_file}')
print('  ✓ POS data loaded')
"

# Step 2: Process videos (Store 1) — using yolov8n for speed
echo ""
echo "═══ Step 2: Processing Store 1 Videos ═══"

# CAM 3 - Entry camera
if [ -f "data/videos/CAM 3 - entry.mp4" ]; then
    echo "  Processing CAM 3 (Entry)..."
    $PYTHON scripts/process_video.py video "data/videos/CAM 3 - entry.mp4" \
        --camera CAM_03 --store STORE_001 \
        --zones config/zones_cam3.json \
        --model yolov8n.pt \
        --no-anomalies || echo "  ⚠ CAM 3 processing had issues, continuing..."
fi

# CAM 1 - Zone camera
if [ -f "data/videos/CAM 1 - zone.mp4" ]; then
    echo "  Processing CAM 1 (Zone)..."
    $PYTHON scripts/process_video.py video "data/videos/CAM 1 - zone.mp4" \
        --camera CAM_01 --store STORE_001 \
        --zones config/zones_cam1.json \
        --model yolov8n.pt \
        --no-anomalies || echo "  ⚠ CAM 1 processing had issues, continuing..."
fi

# CAM 2 - Zone camera
if [ -f "data/videos/CAM 2 - zone.mp4" ]; then
    echo "  Processing CAM 2 (Zone)..."
    $PYTHON scripts/process_video.py video "data/videos/CAM 2 - zone.mp4" \
        --camera CAM_02 --store STORE_001 \
        --zones config/zones_cam2.json \
        --model yolov8n.pt \
        --no-anomalies || echo "  ⚠ CAM 2 processing had issues, continuing..."
fi

# CAM 5 - Billing camera
if [ -f "data/videos/CAM 5 - billing.mp4" ]; then
    echo "  Processing CAM 5 (Billing)..."
    $PYTHON scripts/process_video.py video "data/videos/CAM 5 - billing.mp4" \
        --camera CAM_05 --store STORE_001 \
        --zones config/zones_cam5.json \
        --model yolov8n.pt \
        --no-anomalies || echo "  ⚠ CAM 5 processing had issues, continuing..."
fi

# Step 3: Ingest sample events
echo ""
echo "═══ Step 3: Ingesting Sample Events ═══"
$PYTHON -c "
import sys, json
sys.path.insert(0, '.')
from storage.database import Database
db = Database()
import glob
for jsonl_file in glob.glob('data/new_data/**/*.jsonl', recursive=True):
    events = []
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    result = db.ingest_raw_events(events)
    print(f'  ✓ Ingested {result[\"accepted\"]} events from {jsonl_file} ({result[\"duplicates\"]} duplicates)')
print('  ✓ Sample events loaded')
"

# Step 4: Run anomaly detection
echo ""
echo "═══ Step 4: Running Anomaly Detection ═══"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from storage.database import Database
from analytics.anomaly import AnomalyDetector
db = Database()
detector = AnomalyDetector(db)
detector.compute_baselines('STORE_001')
alerts = detector.run_detection(store_id='STORE_001')
print(f'  ✓ {len(alerts)} anomalies detected')
for a in alerts:
    print(f'    [{a.severity}] {a.anomaly_type}: {a.message}')
"

# Step 5: Print summary
echo ""
echo "═══ Step 5: Summary ═══"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from storage.database import Database
db = Database()
stats = db.get_stats()
print(f'  Events:     {stats[\"total_events\"]}')
print(f'  Journeys:   {stats[\"total_journeys\"]}')
print(f'  Anomalies:  {stats[\"total_anomalies\"]}')
print(f'  POS Txns:   {stats[\"pos_transactions\"]}')
print()
print('  ✓ Pipeline complete!')
print()
print('  Next steps:')
print('    API:       uvicorn api.app:app --reload --port 8000')
print('    Dashboard: streamlit run dashboard/app.py')
print('    Docker:    docker compose up')
"
