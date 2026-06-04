# Store Intelligence — System Design

> AI-Powered Store Intelligence: From raw CCTV footage to actionable retail analytics.

## Repository Structure & Mapping

The project follows a clean, modular structure. While the challenge guidelines suggest a flat layout, we have organized the code into distinct domain directories (`core`, `api`, `storage`, `analytics`, `pipeline`) to separate concerns more cleanly and follow best practices. 

Here is how the suggested layout maps directly to our implementation:

| Suggested Layout File | Our Implementation File | Rationale |
| :--- | :--- | :--- |
| `pipeline/detect.py` | [core/detector.py](file:///Users/vaibhavbansal/store-intelligence/core/detector.py) | Initialises and runs YOLOv8 person detection. |
| `pipeline/tracker.py` | [core/tracker.py](file:///Users/vaibhavbansal/store-intelligence/core/tracker.py) | Manages ByteTrack tracking logic. |
| `pipeline/emit.py` | [core/event_engine.py](file:///Users/vaibhavbansal/store-intelligence/core/event_engine.py) | Handles event creation, formatting, and emission. |
| `pipeline/run.sh` | [run.sh](file:///Users/vaibhavbansal/store-intelligence/run.sh) | Placed in the root directory for easy single-command execution (`./run.sh`). |
| `app/main.py` | [api/app.py](file:///Users/vaibhavbansal/store-intelligence/api/app.py) | The FastAPI entrypoint and router. |
| `app/models.py` | [core/models.py](file:///Users/vaibhavbansal/store-intelligence/core/models.py) | Pydantic event/journey schemas matching Purplle's format. |
| `app/ingestion.py` | [storage/database.py](file:///Users/vaibhavbansal/store-intelligence/storage/database.py) | DB operations for raw event batch ingestion and deduplication. |
| `app/metrics.py` | [core/event_engine.py](file:///Users/vaibhavbansal/store-intelligence/core/event_engine.py) | Computes real-time conversion rate, visitors, and queue analytics. |
| `app/funnel.py` | [core/event_engine.py](file:///Users/vaibhavbansal/store-intelligence/core/event_engine.py) | Reconstructs conversion funnel stages. |
| `app/anomalies.py` | [analytics/anomaly.py](file:///Users/vaibhavbansal/store-intelligence/analytics/anomaly.py) | Detects queue spikes, dead zones, and conversion drops. |
| `app/health.py` | [api/app.py](file:///Users/vaibhavbansal/store-intelligence/api/app.py) | Healthcheck router with stale feed alerts. |
| `docs/DESIGN.md` | [DESIGN.md](file:///Users/vaibhavbansal/store-intelligence/DESIGN.md) | Kept in the root directory as strictly required by Acceptance Gate #5. |
| `docs/CHOICES.md` | [CHOICES.md](file:///Users/vaibhavbansal/store-intelligence/CHOICES.md) | Kept in the root directory as strictly required by Acceptance Gate #5. |

---

## Architecture Overview

The system is a four-stage pipeline that transforms raw CCTV footage into structured business intelligence:

```
┌──────────────────────────────────────────────────────────────┐
│                  Raw CCTV Footage (5 cameras)                │
│   Entry camera │ Zone camera ×2 │ Billing camera             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Stage 1: Detection Layer                        │
│  ┌─────────────┐   ┌───────────┐   ┌──────────────────┐    │
│  │  YOLOv8m    │──▶│ ByteTrack │──▶│  Zone Classifier │    │
│  │  (Person    │   │ (Multi-   │   │  (Polygon-based) │    │
│  │   Detection)│   │  Object   │   │                  │    │
│  └─────────────┘   │  Tracking)│   └──────────┬───────┘    │
│                     └───────────┘              │            │
│                                                │            │
│  Staff Detection: Dwell pattern + movement frequency        │
│  Group Detection: Temporal proximity clustering             │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Stage 2: Event Stream                           │
│                                                              │
│  Three event categories:                                     │
│  1. Entry/Exit events (entry camera)                         │
│     → id_token, is_staff, gender, age, group_id              │
│  2. Zone events (zone cameras)                               │
│     → zone_entered, zone_exited, zone_dwell                  │
│  3. Queue events (billing camera)                            │
│     → queue_joined, queue_completed, queue_abandoned         │
│                                                              │
│  Per-person state machine:                                   │
│  OUTSIDE → ENTRY → IN_ZONE → BILLING_QUEUE → EXIT           │
└────────────────────────────────────────────────┬─────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Stage 3: Intelligence API (FastAPI)             │
│                                                              │
│  POST /events/ingest  — Batch ingest, idempotent             │
│  GET  /stores/{id}/metrics — Real-time store analytics       │
│  GET  /stores/{id}/funnel  — Conversion funnel               │
│  GET  /stores/{id}/heatmap — Zone heat map (0-100)           │
│  GET  /stores/{id}/anomalies — Live anomaly detection        │
│  GET  /health — Service status + STALE_FEED detection        │
│                                                              │
│  Storage: SQLite with WAL mode                               │
│  POS correlation via pos_transactions.csv                    │
└────────────────────────────────────────────────┬─────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Stage 4: Live Dashboard (Streamlit)             │
│                                                              │
│  7 pages: Overview, Funnel, Zones, Journeys, Events,         │
│           Queue Monitor, Anomalies, Live Video Analysis       │
│                                                              │
│  Features: Real-time metrics update, 2x2 camera grid,        │
│            glassmorphism dark theme                            │
└──────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Multi-Camera Architecture
Each camera serves a specific role: entry cameras count visitors, zone cameras track product interest, and billing cameras monitor queue behavior. This mirrors how real retail CCTV systems are deployed. Events from different cameras are unified by `store_id` in the database, enabling cross-camera analytics without requiring cross-camera person re-identification (which is computationally expensive and unreliable with blurred faces).

### Event Schema Design
The event schema was designed around three distinct event categories after analyzing the `sample_events.jsonl` provided. Entry/exit events carry demographic data (gender, age predictions), zone events carry spatial data (zone_id, hotspot coordinates), and queue events carry temporal data (join/serve/exit timestamps). This schema enables the API to answer both real-time operational questions ("What's the queue depth now?") and strategic questions ("Which zones attract customers but don't convert?").

### Staff Detection Strategy
Staff are detected using behavioral heuristics rather than uniform recognition:
1. **Duration-based**: Persons visible for >80% of processed frames are likely staff
2. **Movement pattern**: Staff typically traverse multiple zones frequently
3. **Zone frequency**: Staff enter >3 distinct zones within a session

This approach avoids the need for a separate classifier or VLM, works with blurred faces, and is robust across different store layouts.

### POS Correlation
Conversion rate is computed by correlating `pos_transactions.csv` timestamps with visitor sessions. A visitor in the billing zone within a 5-minute window before a POS transaction is counted as a converted visitor. This is more accurate than pure dwell-time inference.

## AI-Assisted Decisions

### 1. Detection Model Selection (AI shaped → I agreed)
I prompted Claude to compare YOLOv8, YOLOv9, RT-DETR, and MediaPipe for retail person detection with blurred faces. The AI recommended **YOLOv8m** as the optimal balance of speed and accuracy for retail-scale scenes. I agreed because:
- RT-DETR has ~2% better mAP but 3x slower inference — unacceptable for real-time
- YOLOv8m (medium) gives better accuracy than YOLOv8n (nano) at acceptable inference cost
- MediaPipe is designed for close-up pose estimation, not surveillance-distance detection

### 2. Event Schema Design (AI suggested → I modified)
I asked Gemini to design an event schema for retail store analytics. The AI suggested a flat schema with all fields present on every event. I **modified this** to use a category-based approach (entry/exit, zone, queue) because:
- Different cameras produce fundamentally different event types
- A flat schema with many nullable fields wastes storage and makes queries ambiguous
- The sample_events.jsonl confirmed that Purplle uses category-specific fields

### 3. Anomaly Detection Approach (AI suggested → I overrode partially)
ChatGPT suggested using Isolation Forest for anomaly detection. I chose a **hybrid approach** instead:
- Tier 1: Rule-based guardrails (always reliable, no training data needed)
- Tier 2: Z-score statistical detection (self-calibrating from available data)
- I **deferred** Isolation Forest because it requires sufficient historical data for training, which we don't have with only a few hours of footage. In production with weeks of data, Isolation Forest would add multi-dimensional anomaly detection (e.g., normal traffic but abnormal conversion pattern → potential shoplifting).

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Detection | YOLOv8m (Ultralytics) | Best speed/accuracy for retail surveillance |
| Tracking | ByteTrack (supervision) | No re-ID model needed, handles crowds well |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Storage | SQLite (WAL mode) | Zero-setup, sufficient for demo scale |
| Dashboard | Streamlit + Plotly | Rapid development, rich visualization |
| Container | Docker + docker-compose | One-command deployment |

## Edge Case Handling

| Edge Case | How We Handle It |
|-----------|-----------------|
| Group entry | ByteTrack assigns individual IDs; each person gets separate ENTRY event |
| Staff movement | Behavioral heuristics flag `is_staff=true`; excluded from customer metrics |
| Re-entry | Spatial + temporal matching within 5-min window |
| Partial occlusion | Low confidence detections kept (not dropped); flagged with confidence score |
| Queue buildup | Queue position tracked; anomaly alert at depth >5 |
| Empty store | API returns valid zero-valued responses, never null or crashes |
| Camera overlap | Cameras process independently; store-level dedup by person_id prefix |
