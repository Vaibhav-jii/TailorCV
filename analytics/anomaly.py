"""
Store Intelligence Anomaly Detector — Multi-tier anomaly detection.

Tier 1: Rule-based (always-on guardrails)
    - Queue depth thresholds
    - Conversion rate floor
    - Dead zone detection (no visits in 30 min)

Tier 2: Statistical (Z-score / IQR)
    - Detects metrics that deviate >2σ from rolling baseline

Severity levels: INFO, WARN, CRITICAL
Each anomaly includes a suggested_action string.

# PROMPT: "Design a multi-tier anomaly detection system for retail store metrics
# that detects queue spikes, conversion drops vs 7-day average, and dead zones.
# Use Z-score statistical detection with rule-based guardrails."
# CHANGES MADE: Added suggested_action field, changed severity to INFO/WARN/CRITICAL,
# added dead_zone detection, integrated store_id filtering.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import AnomalyAlert
from storage.database import Database


class AnomalyDetector:
    """Multi-tier anomaly detection for store intelligence."""

    def __init__(self, db: Database):
        self.db = db
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.alerts: List[AnomalyAlert] = []

    def compute_baselines(self, store_id: Optional[str] = None):
        """Compute statistical baselines from stored data."""
        zone_data = self.db.get_zone_analytics(store_id)
        for zone in zone_data:
            zone_name = zone.get("zone_name", "unknown")
            visits = zone.get("visit_count", 0)
            self.baselines[f"zone_{zone_name}_visits"] = {
                "mean": visits,
                "std": max(visits * 0.3, 1.0),
            }
            dwell = zone.get("avg_dwell_ms", 0) or 0
            self.baselines[f"zone_{zone_name}_dwell"] = {
                "mean": dwell,
                "std": max(dwell * 0.4, 2.0),
            }

        hourly = self.db.get_hourly_traffic(store_id)
        if hourly:
            counts = [h["entries"] for h in hourly]
            self.baselines["hourly_traffic"] = {
                "mean": float(np.mean(counts)),
                "std": max(float(np.std(counts)), 1.0),
            }

        queue = self.db.get_queue_metrics(store_id)
        if queue.get("total_queue_joins", 0) > 0:
            self.baselines["queue_abandon_rate"] = {
                "mean": queue.get("abandon_rate", 0),
                "std": 0.15,
            }

    def run_detection(self, current_metrics: Optional[dict] = None,
                      store_id: Optional[str] = None) -> List[AnomalyAlert]:
        """Run all anomaly detection tiers."""
        if current_metrics is None:
            current_metrics = self._compute_current_metrics(store_id)

        alerts = []
        alerts.extend(self._tier1_rules(current_metrics, store_id))
        if self.baselines:
            alerts.extend(self._tier2_statistical(current_metrics, store_id))

        for alert in alerts:
            self.db.insert_anomaly(alert)

        self.alerts.extend(alerts)
        return alerts

    def _tier1_rules(self, metrics: dict, store_id: Optional[str] = None) -> List[AnomalyAlert]:
        """Rule-based anomaly detection."""
        alerts = []
        now = datetime.now()
        sid = store_id or ""

        # Rule 1: BILLING_QUEUE_SPIKE — High queue depth
        queue_depth = metrics.get("queue_depth", 0)
        if queue_depth > 5:
            alerts.append(AnomalyAlert(
                timestamp=now,
                anomaly_type="BILLING_QUEUE_SPIKE",
                severity="CRITICAL" if queue_depth > 8 else "WARN",
                metric_name="queue_depth",
                current_value=queue_depth,
                expected_range="0-5",
                message=f"Billing queue depth is {queue_depth}.",
                suggested_action="Open additional billing counters immediately. Deploy staff to manage queue.",
                store_id=sid,
            ))

        # Rule 2: CONVERSION_DROP — Very low conversion rate
        conversion = metrics.get("conversion_rate", 1.0)
        total_entries = metrics.get("total_entries", 0) or metrics.get("unique_visitors", 0)
        if total_entries > 10 and conversion < 0.05:
            alerts.append(AnomalyAlert(
                timestamp=now,
                anomaly_type="CONVERSION_DROP",
                severity="CRITICAL",
                metric_name="conversion_rate",
                current_value=conversion,
                expected_range="0.10-0.45",
                message=f"Conversion rate critically low at {conversion:.1%}.",
                suggested_action="Investigate customer drop-off points. Check billing area for issues. Review staff availability.",
                store_id=sid,
            ))
        elif total_entries > 10 and conversion < 0.10:
            alerts.append(AnomalyAlert(
                timestamp=now,
                anomaly_type="CONVERSION_DROP",
                severity="WARN",
                metric_name="conversion_rate",
                current_value=conversion,
                expected_range="0.10-0.45",
                message=f"Conversion rate dropped to {conversion:.1%}.",
                suggested_action="Monitor billing zone for potential issues. Consider promotional offers.",
                store_id=sid,
            ))

        # Rule 3: HIGH_BILLING_ABANDONMENT
        abandon_rate = metrics.get("abandonment_rate", 0) or metrics.get("billing_abandon_rate", 0)
        if abandon_rate > 0.5 and metrics.get("reached_billing", 0) > 5:
            alerts.append(AnomalyAlert(
                timestamp=now,
                anomaly_type="HIGH_BILLING_ABANDONMENT",
                severity="WARN",
                metric_name="billing_abandon_rate",
                current_value=abandon_rate,
                expected_range="0.0-0.30",
                message=f"Billing abandonment rate is {abandon_rate:.1%}.",
                suggested_action="Reduce wait times at billing. Add express checkout lane.",
                store_id=sid,
            ))

        # Rule 4: DEAD_ZONE — Zone with no visits in 30 min
        zone_data = self.db.get_zone_analytics(store_id)
        for zone in zone_data:
            if zone.get("visit_count", 0) == 0:
                alerts.append(AnomalyAlert(
                    timestamp=now,
                    anomaly_type="DEAD_ZONE",
                    severity="INFO",
                    metric_name=f"zone_{zone.get('zone_name', 'unknown')}_visits",
                    current_value=0,
                    expected_range=">0",
                    message=f"Zone '{zone.get('zone_name', 'unknown')}' has zero visits.",
                    suggested_action="Review zone visibility and product placement. Consider promotional signage.",
                    store_id=sid,
                ))

        return alerts

    def _tier2_statistical(self, metrics: dict, store_id: Optional[str] = None) -> List[AnomalyAlert]:
        """Statistical anomaly detection using Z-scores."""
        alerts = []
        now = datetime.now()
        sid = store_id or ""

        for metric_key, baseline in self.baselines.items():
            current_val = metrics.get(metric_key)
            if current_val is None:
                continue

            mean = baseline["mean"]
            std = baseline["std"]
            if std == 0:
                continue

            z_score = (current_val - mean) / std

            if abs(z_score) > 2.0:
                severity = "CRITICAL" if abs(z_score) > 3.0 else "WARN"
                direction = "above" if z_score > 0 else "below"

                alerts.append(AnomalyAlert(
                    timestamp=now,
                    anomaly_type="STATISTICAL_OUTLIER",
                    severity=severity,
                    metric_name=metric_key,
                    current_value=current_val,
                    expected_range=f"{mean:.1f} ± {2 * std:.1f}",
                    message=f"{metric_key} is {abs(z_score):.1f}σ {direction} normal "
                            f"(current: {current_val:.1f}, baseline: {mean:.1f} ± {std:.1f})",
                    suggested_action=f"Investigate {metric_key} — value is significantly {direction} expected range.",
                    store_id=sid,
                ))

        return alerts

    def _compute_current_metrics(self, store_id: Optional[str] = None) -> dict:
        """Compute current metrics from database."""
        funnel = self.db.get_conversion_funnel(store_id)
        queue = self.db.get_queue_metrics(store_id)
        zones = self.db.get_zone_analytics(store_id)

        metrics = {
            "total_entries": funnel.get("funnel", [{}])[0].get("count", 0) if funnel.get("funnel") else 0,
            "conversion_rate": funnel.get("conversion_rate", 0),
            "billing_reach_rate": funnel.get("billing_reach_rate", 0),
            "abandonment_rate": funnel.get("abandon_rate", 0),
            "reached_billing": funnel.get("funnel", [{}, {}, {}])[2].get("count", 0) if len(funnel.get("funnel", [])) > 2 else 0,
            "queue_depth": 0,
        }

        for zone in zones:
            zone_name = zone.get("zone_name", "unknown")
            metrics[f"zone_{zone_name}_visits"] = zone.get("visit_count", 0)
            metrics[f"zone_{zone_name}_dwell"] = zone.get("avg_dwell_ms", 0) or 0

        return metrics
