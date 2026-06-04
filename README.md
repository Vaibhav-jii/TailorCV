# Store Intelligence 🏪

AI-powered retail intelligence from CCTV footage, built for the Purplle Tech Challenge.

## Quick Start (Evaluation Ready)

The entire system is containerized. Start it with a single command:

```bash
docker compose up --build
```

This starts:
1. **Intelligence API**: `http://localhost:8000` (Swagger UI at `/docs`)
2. **Live Dashboard**: `http://localhost:8501`

## Running the Pipeline

To process the raw CCTV video clips, correlate them with POS transaction history, and run anomaly detection, execute the pipeline runner script:

```bash
./run.sh
```

### Where the Output Goes

The detection pipeline processes the clips and routes output to the following locations:
1. **SQLite Database**: Saved to `data/store_intelligence.db`. This file contains the schema for all processed events, calculated journeys, detected anomalies, and loaded POS transactions.
2. **API Layer**: Data is served via real-time endpoints at `http://localhost:8000`:
   - `GET /stores/STORE_001/metrics` — Core store metrics (conversion rate, unique visitors, queue depth)
   - `GET /stores/STORE_001/funnel` — Conversion funnel (Entry → Zone → Queue → Purchase)
   - `GET /stores/STORE_001/heatmap` — Heatmap coordinates normalised 0-100
   - `GET /stores/STORE_001/anomalies` — Alerts (e.g. conversion drops, queue builds)
3. **Interactive Dashboard**: Accessible at `http://localhost:8501` to visualize dwell times, traffic heatmaps, and funnel drop-offs.
4. **Annotated Videos & Heatmaps**: Frame-by-frame visual outputs and heatmap plots are generated and saved to `data/output/` (if enabled in `process_video.py`).

## Challenge Requirements Fulfilled

✅ **Dockerized**: `docker compose up` starts everything
✅ **Required Endpoints**: `POST /events/ingest`, `GET /stores/{id}/metrics`, `/funnel`, `/heatmap`, `/anomalies`, `/health`
✅ **Schema Matched**: Uses exact fields from `sample_events.jsonl` (UUIDs, `is_staff`, `dwell_ms`, `queue_depth`, etc.)
✅ **Staff Detection**: Behavioral heuristic deployed (excludes staff from customer metrics)
✅ **POS Correlation**: Converts billing presence into confirmed purchases via 5-min window correlation
✅ **Documentation**: `DESIGN.md` and `CHOICES.md` included
✅ **Tests**: Pytest suite covering all endpoints, idempotency, edge cases, and >70% coverage

## Architecture & Choices

Please see:
- [DESIGN.md](DESIGN.md) for system architecture and AI-assisted decisions.
- [CHOICES.md](CHOICES.md) for detailed technical rationale on model selection and API design.
