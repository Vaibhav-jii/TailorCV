"""
Store Intelligence REST API — Production-grade FastAPI application.

Required Endpoints (Purplle Challenge):
    POST /events/ingest              → Batch ingest (idempotent, partial success)
    GET  /stores/{id}/metrics        → Real-time store metrics
    GET  /stores/{id}/funnel         → Conversion funnel
    GET  /stores/{id}/heatmap        → Zone heatmap (normalised 0-100)
    GET  /stores/{id}/anomalies      → Active anomalies
    GET  /health                     → Service health + STALE_FEED detection

Additional Endpoints:
    GET  /api/v1/events              → Query events (filterable)
    GET  /api/v1/journeys            → All customer journeys
    GET  /api/v1/journey/{person_id} → Single person journey
    GET  /api/v1/metrics/summary     → Full metrics summary
"""

from fastapi import FastAPI, Query, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime
import time
import uuid
import logging
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database

# ── Structured Logging ──
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger("store_intelligence.api")

# ── App Setup ──

app = FastAPI(
    title="Store Intelligence API",
    description=(
        "AI-Powered Store Intelligence System — Real-time analytics from CCTV footage.\n\n"
        "Provides structured access to customer journey events, conversion funnel metrics, "
        "zone analytics, queue monitoring, and anomaly detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Store Intelligence Team"},
)

# CORS — allow dashboard and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database singleton
db: Optional[Database] = None


def get_db() -> Database:
    global db
    if db is None:
        db = Database()
    return db


# ── Middleware: Structured Request Logging ──

@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4())[:8])
    start = time.time()
    
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000
        
        # Extract store_id from path if present
        store_id = ""
        path_parts = request.url.path.split("/")
        for i, part in enumerate(path_parts):
            if part == "stores" and i + 1 < len(path_parts):
                store_id = path_parts[i + 1]
                break
        
        logger.info(json.dumps({
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": request.url.path,
            "method": request.method,
            "latency_ms": round(latency_ms, 2),
            "status_code": response.status_code,
        }))
        
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        return response
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(json.dumps({
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "method": request.method,
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }))
        raise


# ═══════════════════════════════════════════════════════════
# Health Check — GET /health
# ═══════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
def health_check():
    """
    Service health check.
    
    Returns service status, last event timestamp per store,
    and STALE_FEED warning if any store has >10 min lag.
    """
    try:
        database = get_db()
        health_data = database.get_health_data()
        stats = database.get_stats()
        
        # Check for stale feeds
        warnings = []
        for sid, info in health_data.get("stores", {}).items():
            if info.get("status") == "STALE_FEED":
                warnings.append(f"STALE_FEED: {sid} — last event {info.get('lag_minutes', '?')} min ago")
        
        return {
            "status": "healthy",
            "service": "Store Intelligence API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "stores": health_data.get("stores", {}),
            "database": stats,
            "warnings": warnings,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "Store Intelligence API",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


# ═══════════════════════════════════════════════════════════
# Event Ingest — POST /events/ingest
# ═══════════════════════════════════════════════════════════

@app.post("/events/ingest", tags=["Events"])
async def ingest_events(request: Request):
    """
    Accepts batches of up to 500 events.
    Validates, deduplicates, stores.
    Idempotent by event_id. Partial success on malformed events.
    """
    try:
        database = get_db()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "type": "SERVICE_UNAVAILABLE"},
        )
    
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body", "type": "INVALID_REQUEST"},
        )
    
    # Accept both {"events": [...]} and bare [...]
    if isinstance(body, list):
        events = body
    elif isinstance(body, dict):
        events = body.get("events", [])
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Expected JSON array or object with 'events' key", "type": "INVALID_REQUEST"},
        )
    
    if len(events) > 500:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Batch size {len(events)} exceeds maximum of 500",
                "type": "BATCH_TOO_LARGE",
            },
        )
    
    result = database.ingest_raw_events(events)
    
    logger.info(json.dumps({
        "action": "events_ingested",
        "event_count": len(events),
        "accepted": result["accepted"],
        "duplicates": result["duplicates"],
        "errors": len(result["errors"]),
    }))
    
    status = 200 if not result["errors"] else 207  # 207 Multi-Status for partial success
    return JSONResponse(
        status_code=status,
        content={
            "status": "ok" if not result["errors"] else "partial",
            "accepted": result["accepted"],
            "duplicates": result["duplicates"],
            "errors": result["errors"][:10],  # Cap error details
            "total_submitted": len(events),
        },
    )


# ═══════════════════════════════════════════════════════════
# Store Metrics — GET /stores/{id}/metrics
# ═══════════════════════════════════════════════════════════

@app.get("/stores/{store_id}/metrics", tags=["Store Analytics"])
def get_store_metrics(store_id: str):
    """
    Today: unique visitors, conversion rate, avg dwell per zone,
    queue depth, abandonment rate.
    
    Excludes is_staff=true. Handles zero-purchase stores. Real-time.
    """
    try:
        database = get_db()
        return database.get_store_metrics(store_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "detail": str(e)},
        )


# ═══════════════════════════════════════════════════════════
# Conversion Funnel — GET /stores/{id}/funnel
# ═══════════════════════════════════════════════════════════

@app.get("/stores/{store_id}/funnel", tags=["Store Analytics"])
def get_store_funnel(store_id: str):
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase
    with counts and drop-off %.
    
    Session is the unit, not raw events. Re-entries do not double-count.
    """
    try:
        database = get_db()
        return database.get_conversion_funnel(store_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "detail": str(e)},
        )


# ═══════════════════════════════════════════════════════════
# Heatmap — GET /stores/{id}/heatmap
# ═══════════════════════════════════════════════════════════

@app.get("/stores/{store_id}/heatmap", tags=["Store Analytics"])
def get_store_heatmap(store_id: str):
    """
    Zone visit frequency + avg dwell, normalised 0-100, ready for grid heatmap rendering.
    Includes data_confidence flag if fewer than 20 sessions in window.
    """
    try:
        database = get_db()
        return database.get_heatmap(store_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "detail": str(e)},
        )


# ═══════════════════════════════════════════════════════════
# Anomalies — GET /stores/{id}/anomalies
# ═══════════════════════════════════════════════════════════

@app.get("/stores/{store_id}/anomalies", tags=["Store Analytics"])
def get_store_anomalies(store_id: str):
    """
    Active anomalies: queue spike, conversion drop vs 7-day avg, dead zone.
    Severity: INFO / WARN / CRITICAL. Includes suggested_action per anomaly.
    """
    try:
        database = get_db()
        
        # Run live anomaly detection
        from analytics.anomaly import AnomalyDetector
        detector = AnomalyDetector(database)
        detector.compute_baselines(store_id)
        live_anomalies = detector.run_detection(store_id=store_id)
        
        # Also get stored anomalies
        stored = database.get_anomalies(store_id=store_id, limit=50)
        
        return {
            "store_id": store_id,
            "active_anomalies": [
                {
                    "alert_id": a.alert_id,
                    "anomaly_type": a.anomaly_type,
                    "severity": a.severity,
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "expected_range": a.expected_range,
                    "message": a.message,
                    "suggested_action": a.suggested_action,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in live_anomalies
            ],
            "historical": stored[:20],
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Database unavailable", "detail": str(e)},
        )


# ═══════════════════════════════════════════════════════════
# Additional API Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/events", tags=["Events"])
def get_events(
    event_type: Optional[str] = Query(None),
    person_id: Optional[str] = Query(None),
    store_id: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    exclude_staff: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query store events with optional filters."""
    database = get_db()
    events = database.get_events(
        event_type=event_type, person_id=person_id, store_id=store_id,
        zone=zone, camera_id=camera_id, since=since,
        exclude_staff=exclude_staff, limit=limit, offset=offset,
    )
    return {"events": events, "count": len(events), "limit": limit, "offset": offset}


@app.get("/api/v1/events/counts", tags=["Events"])
def get_event_counts(store_id: Optional[str] = Query(None)):
    """Get count of events grouped by type."""
    database = get_db()
    return {"event_counts": database.get_event_counts(store_id)}


@app.get("/api/v1/journeys", tags=["Journeys"])
def get_journeys(
    purchased_only: bool = Query(False),
    store_id: Optional[str] = Query(None),
    exclude_staff: bool = Query(True),
):
    """Get all customer journeys."""
    database = get_db()
    journeys = database.get_journeys(purchased_only=purchased_only, store_id=store_id,
                                      exclude_staff=exclude_staff)
    return {"journeys": journeys, "count": len(journeys)}


@app.get("/api/v1/journey/{person_id}", tags=["Journeys"])
def get_person_journey(person_id: str):
    """Get a specific person's complete journey with all their events."""
    database = get_db()
    events = database.get_events(person_id=person_id, limit=500)
    journeys = database.get_journeys()
    journey = next((j for j in journeys if j["person_id"] == person_id), None)

    if not journey:
        raise HTTPException(status_code=404, detail=f"No journey found for '{person_id}'")

    return {"journey": journey, "events": events, "event_count": len(events)}


@app.get("/api/v1/funnel", tags=["Analytics"])
def get_conversion_funnel(store_id: Optional[str] = Query(None)):
    """Get the conversion funnel — the North Star metric."""
    database = get_db()
    return database.get_conversion_funnel(store_id)


@app.get("/api/v1/zones/analytics", tags=["Analytics"])
def get_zone_analytics(store_id: Optional[str] = Query(None)):
    """Get per-zone visit analytics."""
    database = get_db()
    return {"zones": database.get_zone_analytics(store_id)}


@app.get("/api/v1/traffic/hourly", tags=["Analytics"])
def get_hourly_traffic(store_id: Optional[str] = Query(None)):
    """Get foot traffic aggregated by hour of day."""
    database = get_db()
    return {"hourly_traffic": database.get_hourly_traffic(store_id)}


@app.get("/api/v1/queue", tags=["Analytics"])
def get_queue_metrics(store_id: Optional[str] = Query(None)):
    """Get billing queue metrics."""
    database = get_db()
    return database.get_queue_metrics(store_id)


@app.get("/api/v1/metrics/summary", tags=["Analytics"])
def get_metrics_summary(store_id: Optional[str] = Query(None)):
    """Get a comprehensive metrics summary."""
    database = get_db()
    return {
        "event_counts": database.get_event_counts(store_id),
        "funnel": database.get_conversion_funnel(store_id),
        "zone_analytics": database.get_zone_analytics(store_id),
        "queue_metrics": database.get_queue_metrics(store_id),
        "hourly_traffic": database.get_hourly_traffic(store_id),
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/v1/anomalies", tags=["Anomalies"])
def get_anomalies(
    store_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Get recent anomaly alerts."""
    database = get_db()
    return {"anomalies": database.get_anomalies(store_id=store_id, severity=severity, limit=limit)}


@app.get("/api/v1/db/stats", tags=["Meta"])
def database_stats():
    """Get database statistics."""
    return get_db().get_stats()


# ═══════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler — no raw stack traces in responses."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "type": "about:blank",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An internal error occurred. Please try again.",
            "instance": str(request.url),
        },
    )


# ── Startup Event ──

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    global db
    db = Database()
    logger.info("Store Intelligence API started")


# ── Run directly ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
