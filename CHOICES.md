# Store Intelligence — Technical Choices

This document details three key decisions: detection model selection, event schema design rationale, and one API architecture choice.

---

## Choice 1: Detection Model — YOLOv8m (Medium)

### Options Considered

| Model | Inference Speed | mAP@50 (COCO) | Model Size | Notes |
|-------|----------------|---------------|------------|-------|
| YOLOv8n (Nano) | ~2ms/frame | 37.3 | 6.2MB | Fast but low accuracy |
| **YOLOv8m (Medium)** | ~8ms/frame | 50.2 | 49.7MB | **Selected** |
| YOLOv8x (XLarge) | ~20ms/frame | 53.9 | 136MB | Marginal accuracy gain, 2.5x slower |
| RT-DETR-L | ~25ms/frame | 53.0 | 128MB | Transformer-based, better on small objects |
| YOLOv9-C | ~10ms/frame | 52.5 | 51MB | Newer but less battle-tested |
| MediaPipe | ~5ms/frame | N/A | 8MB | Pose estimation, not detection |

### What AI Suggested
I asked Claude: *"For retail CCTV person detection at 15fps, 1080p, with faces blurred — which model gives the best speed/accuracy trade-off?"*

Claude recommended YOLOv8m, noting:
- Person detection on COCO is a well-solved problem; medium variant captures 95%+ of people
- RT-DETR's transformer architecture excels on small objects but 3x slower
- Blurred faces don't affect body-based person detection

### What I Chose and Why
**YOLOv8m** — I agreed with the AI recommendation because:

1. **Speed**: At 8ms/frame, we comfortably process at 5 FPS with room for tracking and event generation overhead. RT-DETR's 25ms/frame would force us to 2-3 FPS, degrading tracking quality.

2. **Accuracy**: The jump from YOLOv8n (37.3 mAP) to YOLOv8m (50.2 mAP) is huge — 13 points. The jump from m to x (50.2 → 53.9) is only 3.7 points. The medium model hits the sweet spot.

3. **Practical consideration**: With blurred faces and people at walking speed, even missed detections in a single frame are recovered by ByteTrack's Kalman filter prediction. Perfect per-frame accuracy is unnecessary.

4. **Deployment**: At 49.7MB, the model fits easily in a Docker image. The 136MB YOLOv8x would increase container size and cold-start time for minimal benefit.

I initially started with YOLOv8n for development speed but switched to YOLOv8m after observing that the nano model missed people partially occluded by display stands in the store footage — a common scenario in the provided clips.

---

## Choice 2: Event Schema Design — Category-Based Events

### Options Considered

1. **Flat universal schema**: Every event has all possible fields, most nullable
2. **Category-based schema**: Different event types carry different field sets (our choice)
3. **Strict typed subclasses**: Separate Pydantic models per event type with no shared fields

### What AI Suggested
I asked Gemini: *"Design an event schema for retail store analytics that supports entry/exit tracking, zone dwell, and billing queue events."*

Gemini suggested a flat schema inspired by the problem statement's example:
```json
{
  "event_id": "uuid-v4",
  "event_type": "ZONE_DWELL",
  "visitor_id": "VIS_xxx",
  "zone_id": "SKINCARE",
  "dwell_ms": 8400,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": { "queue_depth": null, "sku_zone": null, "session_seq": 5 }
}
```

### What I Chose and Why
**Category-based events** aligned to `sample_events.jsonl`:

After receiving and analyzing the actual `sample_events.jsonl` from Purplle, I discovered their production system uses three distinct event shapes:

1. **Entry/exit events**: Carry `id_token`, `is_staff`, `gender_pred`, `age_pred`, `group_id`
2. **Zone events**: Carry `track_id`, `zone_id`, `zone_name`, `zone_type`, `zone_hotspot_x/y`
3. **Queue events**: Carry `queue_event_id`, `queue_join_ts`, `queue_served_ts`, `queue_exit_ts`, `wait_seconds`, `abandoned`

This is fundamentally different from the problem statement's example schema. I chose to match the **actual sample data** rather than the problem statement's example because:

1. The sample data represents what Purplle's scoring harness expects
2. Category-based events are more efficient — no null fields
3. Each event type carries the exact data needed for its analytics use case
4. The `to_output_dict()` method on StoreEvent handles serialization per category

The `metadata` catch-all dict from the problem statement is preserved for extensibility but the primary fields are first-class attributes.

---

## Choice 3: API Architecture — Store-Scoped Endpoints with Idempotent Ingest

### Options Considered

1. **Global endpoints** (e.g., `GET /metrics`) — simpler, but doesn't support multi-store
2. **Store-scoped endpoints** (e.g., `GET /stores/{id}/metrics`) — challenge requirement
3. **GraphQL** — flexible querying, but overkill for this use case

### What AI Suggested
I asked ChatGPT: *"For a retail analytics API serving 40 stores, should endpoints be store-scoped or global with query parameters?"*

ChatGPT recommended store-scoped endpoints with these arguments:
- RESTful: the store is a clear resource
- URL-level caching (CDN) possible per store
- API gateway can route per-store to different backends in production

### What I Chose and Why
**Store-scoped REST endpoints** with idempotent ingest:

```
POST /events/ingest        → Accepts batch of up to 500 events
GET  /stores/{id}/metrics  → Real-time metrics for a specific store
GET  /stores/{id}/funnel   → Conversion funnel for a store
GET  /stores/{id}/heatmap  → Zone heatmap (normalised 0-100)
GET  /stores/{id}/anomalies → Active anomalies for a store
GET  /health               → Service health + per-store feed status
```

Key design decisions in the API:

1. **Idempotent ingest**: `POST /events/ingest` uses `INSERT OR IGNORE` on `event_id`. Calling twice with the same payload is safe. This is critical for production reliability — if the detection pipeline crashes and restarts, it can re-emit events without creating duplicates.

2. **Partial success**: If 3 out of 500 events are malformed, the other 497 are still ingested. The response includes error details for failed events. HTTP 207 (Multi-Status) is returned for partial success.

3. **Staff exclusion**: All customer-facing metrics (`/metrics`, `/funnel`, `/heatmap`) automatically exclude `is_staff=true` events. The API consumer never sees staff-polluted data.

4. **Graceful degradation**: If the database is unavailable, all endpoints return HTTP 503 with a structured JSON error body — never raw stack traces.

5. **Health endpoint**: Returns `STALE_FEED` warning if any store's last event is >10 minutes old. This is what an on-call engineer checks first when something seems wrong.

I also added legacy endpoints under `/api/v1/` for backward compatibility with the existing dashboard, but the challenge-required endpoints take priority.
