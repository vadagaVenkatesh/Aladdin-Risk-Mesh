"""
ops/monitor.py - Operational Monitoring, Limit Breach Logging & Escalation
============================================================================
Copyright (c) 2026 VDG Venkatesh. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL. Unauthorized use, reproduction or distribution
of this software, in whole or in part, without explicit written permission
from VDG Venkatesh is strictly prohibited.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ops.monitor")


# ---------------------------------------------------------------------------
# Enums & Schemas
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    BREACH = "BREACH"
    CRITICAL = "CRITICAL"


class BreachType(str, Enum):
    VAR = "VaR"
    STRESS_LOSS = "StressLoss"
    GROSS_EXPOSURE = "GrossExposure"
    NET_EXPOSURE = "NetExposure"
    CONCENTRATION = "Concentration"
    DRAWDOWN = "Drawdown"
    LEVERAGE = "Leverage"
    DATA_QUALITY = "DataQuality"
    MODEL_FAILURE = "ModelFailure"
    EXECUTION = "Execution"


@dataclass
class LimitBreachEvent:
    """Schema for a single limit breach event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    breach_type: str = BreachType.VAR
    severity: str = Severity.BREACH
    strategy: str = "global"
    desk: str = "all"
    metric_name: str = ""
    metric_value: float = 0.0
    limit_value: float = 0.0
    breach_pct: float = 0.0          # how far over the limit (%)
    context: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution_note: str = ""


@dataclass
class IncidentEvent:
    """Schema for an operational incident (data, model, execution)."""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    incident_type: str = ""
    severity: str = Severity.WARNING
    description: str = ""
    affected_modules: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    resolved: bool = False


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------
class OperationalMonitor:
    """
    Central operational monitor for the Aladdin-Risk-Mesh platform.

    Responsibilities
    ----------------
    - Record and persist limit breach events and operational incidents.
    - Flag breaches against configurable risk limits.
    - Provide query interfaces for dashboard / reporting consumption.
    - Support middle-office escalation workflows.

    Usage
    -----
        monitor = OperationalMonitor(log_dir="logs/")
        monitor.check_var_limit(var_value=0.025, var_limit=0.02, strategy="macro")
        monitor.log_incident("DataQuality", "Missing CPI data for 3 days")
        report = monitor.daily_summary()
    """

    def __init__(self, log_dir: str = "logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._breaches: List[LimitBreachEvent] = []
        self._incidents: List[IncidentEvent] = []
        self._breach_log_path = self.log_dir / "breach_log.jsonl"
        self._incident_log_path = self.log_dir / "incident_log.jsonl"

    # ------------------------------------------------------------------
    # Limit checkers
    # ------------------------------------------------------------------
    def check_var_limit(
        self,
        var_value: float,
        var_limit: float,
        strategy: str = "global",
        desk: str = "all",
    ) -> Optional[LimitBreachEvent]:
        if var_value > var_limit:
            breach_pct = (var_value - var_limit) / var_limit * 100
            severity = Severity.CRITICAL if breach_pct > 20 else Severity.BREACH
            event = LimitBreachEvent(
                breach_type=BreachType.VAR,
                severity=severity,
                strategy=strategy,
                desk=desk,
                metric_name="99%_VaR",
                metric_value=round(var_value, 6),
                limit_value=round(var_limit, 6),
                breach_pct=round(breach_pct, 2),
            )
            self._record_breach(event)
            return event
        return None

    def check_drawdown_limit(
        self,
        drawdown: float,
        drawdown_limit: float,
        strategy: str = "global",
    ) -> Optional[LimitBreachEvent]:
        if abs(drawdown) > drawdown_limit:
            breach_pct = (abs(drawdown) - drawdown_limit) / drawdown_limit * 100
            event = LimitBreachEvent(
                breach_type=BreachType.DRAWDOWN,
                severity=Severity.CRITICAL if breach_pct > 30 else Severity.BREACH,
                strategy=strategy,
                metric_name="MaxDrawdown",
                metric_value=round(drawdown, 6),
                limit_value=round(drawdown_limit, 6),
                breach_pct=round(breach_pct, 2),
            )
            self._record_breach(event)
            return event
        return None

    def check_exposure_limits(
        self,
        gross: float,
        net: float,
        gross_limit: float,
        net_limit: float,
        strategy: str = "global",
    ) -> List[LimitBreachEvent]:
        events = []
        for val, lim, name, btype in [
            (gross, gross_limit, "GrossExposure", BreachType.GROSS_EXPOSURE),
            (abs(net), net_limit, "NetExposure", BreachType.NET_EXPOSURE),
        ]:
            if val > lim:
                breach_pct = (val - lim) / lim * 100
                event = LimitBreachEvent(
                    breach_type=btype,
                    severity=Severity.BREACH,
                    strategy=strategy,
                    metric_name=name,
                    metric_value=round(val, 4),
                    limit_value=round(lim, 4),
                    breach_pct=round(breach_pct, 2),
                )
                self._record_breach(event)
                events.append(event)
        return events

    # ------------------------------------------------------------------
    # Incident logging
    # ------------------------------------------------------------------
    def log_incident(
        self,
        incident_type: str,
        description: str,
        severity: str = Severity.WARNING,
        affected_modules: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> IncidentEvent:
        event = IncidentEvent(
            incident_type=incident_type,
            severity=severity,
            description=description,
            affected_modules=affected_modules or [],
            context=context or {},
        )
        self._incidents.append(event)
        self._append_jsonl(self._incident_log_path, asdict(event))
        logger.warning("INCIDENT [%s] %s: %s", severity, incident_type, description)
        return event

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve_breach(self, event_id: str, note: str) -> bool:
        for b in self._breaches:
            if b.event_id == event_id:
                b.resolved = True
                b.resolution_note = note
                logger.info("Breach %s resolved: %s", event_id, note)
                return True
        return False

    # ------------------------------------------------------------------
    # Reporting & queries
    # ------------------------------------------------------------------
    def daily_summary(self, date: Optional[str] = None) -> Dict:
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        day_breaches = [
            b for b in self._breaches if b.timestamp.startswith(date)
        ]
        day_incidents = [
            i for i in self._incidents if i.timestamp.startswith(date)
        ]
        return {
            "date": date,
            "total_breaches": len(day_breaches),
            "unresolved_breaches": sum(1 for b in day_breaches if not b.resolved),
            "critical_breaches": sum(
                1 for b in day_breaches if b.severity == Severity.CRITICAL
            ),
            "total_incidents": len(day_incidents),
            "breach_types": list({b.breach_type for b in day_breaches}),
            "breaches": [asdict(b) for b in day_breaches],
            "incidents": [asdict(i) for i in day_incidents],
        }

    def breach_history_df(self) -> pd.DataFrame:
        if not self._breaches:
            return pd.DataFrame()
        return pd.DataFrame([asdict(b) for b in self._breaches])

    def incident_history_df(self) -> pd.DataFrame:
        if not self._incidents:
            return pd.DataFrame()
        return pd.DataFrame([asdict(i) for i in self._incidents])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_breach(self, event: LimitBreachEvent):
        self._breaches.append(event)
        self._append_jsonl(self._breach_log_path, asdict(event))
        logger.warning(
            "LIMIT BREACH [%s] %s: %.4f > limit %.4f (%.1f%% over) | strategy=%s",
            event.severity,
            event.metric_name,
            event.metric_value,
            event.limit_value,
            event.breach_pct,
            event.strategy,
        )

    @staticmethod
    def _append_jsonl(path: Path, record: Dict):
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    monitor = OperationalMonitor(log_dir="/tmp/aladdin_logs")

    # Simulate a VaR breach
    monitor.check_var_limit(var_value=0.028, var_limit=0.020, strategy="global_macro")

    # Simulate a drawdown breach
    monitor.check_drawdown_limit(drawdown=-0.12, drawdown_limit=0.10, strategy="carry")

    # Simulate exposure breaches
    monitor.check_exposure_limits(
        gross=3.8, net=0.85, gross_limit=3.0, net_limit=0.5, strategy="ls_equity"
    )

    # Log an incident
    monitor.log_incident(
        "DataQuality",
        "Stale CPI print detected - 3 missing days in FRED feed",
        severity=Severity.WARNING,
        affected_modules=["data", "signals"],
    )

    # Daily summary
    summary = monitor.daily_summary()
    print("\n=== Daily Ops Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== Breach History ===")
    print(monitor.breach_history_df().to_string())
