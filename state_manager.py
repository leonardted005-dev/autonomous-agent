"""
State Manager Module
====================
Provides an explicit Finite State Machine (FSM) tailored for stateless cloud runner environments
such as GitHub Actions, AWS Lambda, or cron container workers.

The agent transitions sequentially through lifecycle states:
    WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP (or ERROR / PENDING_APPROVAL)

Each state change is validated against allowed transitions to prevent invalid execution paths
and ensure complete determinism across ephemeral worker instances.

Non-Blocking Multi-Task Proposal Ledger
---------------------------------------
Requesting human approval no longer freezes the whole lifecycle. Instead of holding a
single blocking `pending_proposal`, the context maintains a persistent global array named
`proposals_ledger`. When a task needs human intervention, an account wall, or external
publishing, it is appended to the ledger with a unique ID and a status of
`PENDING_APPROVAL`; the agent immediately continues its remaining pipeline steps within
the same run. The FSM only settles into SLEEP once its entire multi-step plan for the
cycle has concluded, so many verifications can wait on the dashboard simultaneously while
the agent keeps exploring new opportunities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("StateManager")


class AgentState(str, Enum):
    """Lifecycle states of the autonomous agent execution cycle."""
    WAKE_UP = "WAKE_UP"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    SLEEP = "SLEEP"
    ERROR = "ERROR"


# Strict map of allowed directional transitions between states.
# PENDING_APPROVAL is entered from PLAN/EXECUTE when a task touches external
# platforms, financial transactions, or asset publishing (Human-in-the-Loop gate).
#
# With the Non-Blocking Multi-Task Proposal Ledger the gate is advisory only: a cycle
# may park tasks in PENDING_APPROVAL and keep flowing through EXECUTE -> EVALUATE -> SLEEP
# without halting, so PENDING_APPROVAL is reachable from every working state and may
# return to any of them.
ALLOWED_TRANSITIONS: Dict[AgentState, List[AgentState]] = {
    AgentState.WAKE_UP: [AgentState.PLAN, AgentState.PENDING_APPROVAL, AgentState.ERROR, AgentState.SLEEP],
    AgentState.PLAN: [AgentState.EXECUTE, AgentState.EVALUATE, AgentState.PENDING_APPROVAL, AgentState.ERROR, AgentState.SLEEP],
    AgentState.EXECUTE: [AgentState.EVALUATE, AgentState.PLAN, AgentState.PENDING_APPROVAL, AgentState.ERROR, AgentState.SLEEP],
    AgentState.EVALUATE: [AgentState.PLAN, AgentState.EXECUTE, AgentState.PENDING_APPROVAL, AgentState.SLEEP, AgentState.ERROR],
    AgentState.PENDING_APPROVAL: [AgentState.PLAN, AgentState.EXECUTE, AgentState.EVALUATE, AgentState.SLEEP, AgentState.ERROR, AgentState.WAKE_UP],
    AgentState.SLEEP: [AgentState.WAKE_UP],
    AgentState.ERROR: [AgentState.SLEEP],  # Graceful wind-down after failure logging
}


class ProposalStatus(str, Enum):
    """Lifecycle of a single entry in the global proposals_ledger."""
    PENDING_APPROVAL = "PENDING_APPROVAL"
    COMPLETED = "COMPLETED"
    DENIED = "DENIED"


# Backwards-compatible alias retained for callers importing the plural name.
ProposalState = ProposalStatus


class VentureStatus(str, Enum):
    """
    Lifecycle of a single entry in the global ventures_ledger.

    Every product, program, listing, or micro-business the agent launches is
    registered as a venture so its commercial performance can be tracked across
    stateless runs. `PROFITABLE`, `LOSING`, and `NO_SALES` are derived health
    signals recomputed on every metric snapshot; the rest are explicit states.
    """
    PLANNING = "PLANNING"
    BUILDING = "BUILDING"
    LIVE = "LIVE"
    MARKETING = "MARKETING"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"
    # Derived health markers (never set directly by callers, but valid to persist).
    PROFITABLE = "PROFITABLE"
    LOSING = "LOSING"
    NO_SALES = "NO_SALES"
    BREAKEVEN = "BREAKEVEN"


@dataclass
class StateTransitionRecord:
    """Historical record of an individual state transition."""
    from_state: AgentState
    to_state: AgentState
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """
    Serializable container for ephemeral run context.
    Persisted across state transitions and restored from the memory bridge at wake-up.
    """
    run_id: str
    cycle_count: int = 0
    current_state: AgentState = AgentState.WAKE_UP
    payload: Dict[str, Any] = field(default_factory=dict)
    active_plan: List[str] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    # Persistent global multi-task ledger. Replaces the old single blocking
    # `pending_proposal` dictionary: any number of tasks may wait here for human
    # verification while the agent keeps executing the rest of its pipeline.
    proposals_ledger: List[Dict[str, Any]] = field(default_factory=list)
    # Persistent global venture/portfolio ledger. Every product, program, listing,
    # or micro-business the agent launches is registered here with its commercial
    # metrics (revenue, costs, units sold, visitors) so performance is tracked
    # across stateless runs. Deliberately survives `reset_for_new_cycle()`.
    ventures_ledger: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes context to a plain JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "cycle_count": self.cycle_count,
            "current_state": self.current_state.value,
            "payload": self.payload,
            "active_plan": self.active_plan,
            "execution_results": self.execution_results,
            "working_memory": self.working_memory,
            "errors": self.errors,
            "history": self.history,
            "proposals_ledger": self.proposals_ledger,
            "ventures_ledger": self.ventures_ledger,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentContext:
        """Hydrates context instance from a dictionary."""
        raw_state = data.get("current_state", AgentState.WAKE_UP.value)
        try:
            current_state = AgentState(raw_state)
        except ValueError:
            current_state = AgentState.WAKE_UP

        raw_ledger = data.get("proposals_ledger", [])
        ledger = (
            [dict(entry) for entry in raw_ledger if isinstance(entry, dict)]
            if isinstance(raw_ledger, list) else []
        )

        raw_ventures = data.get("ventures_ledger", [])
        ventures = (
            [dict(entry) for entry in raw_ventures if isinstance(entry, dict)]
            if isinstance(raw_ventures, list) else []
        )

        return cls(
            run_id=data.get("run_id", "unknown_run"),
            cycle_count=data.get("cycle_count", 0),
            current_state=current_state,
            payload=data.get("payload", {}),
            active_plan=data.get("active_plan", []),
            execution_results=data.get("execution_results", []),
            working_memory=data.get("working_memory", []),
            errors=data.get("errors", []),
            history=data.get("history", []),
            proposals_ledger=ledger,
            ventures_ledger=ventures,
        )


class StateTransitionError(Exception):
    """Raised when an illegal or unsupported state transition is attempted."""
    pass


class StateManager:
    """
    Finite State Machine manager controlling agent execution states.
    Validates, records, and guarantees orderly transition flow.
    """

    def __init__(self, context: Optional[AgentContext] = None) -> None:
        if context is None:
            context = AgentContext(run_id=f"run_{int(datetime.now(timezone.utc).timestamp())}")
        self.context = context
        logger.info(
            "StateManager initialized at state: %s (run_id: %s)",
            self.context.current_state.value,
            self.context.run_id,
        )

    @property
    def current_state(self) -> AgentState:
        return self.context.current_state

    def can_transition_to(self, target_state: AgentState) -> bool:
        """Checks if a transition to target_state is allowed from the current state."""
        allowed = ALLOWED_TRANSITIONS.get(self.context.current_state, [])
        return target_state in allowed

    def transition_to(
        self,
        target_state: AgentState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """
        Transitions the FSM to target_state if valid.
        Logs the change and appends to context history.
        """
        if not self.can_transition_to(target_state):
            err_msg = (
                f"Invalid transition attempted: {self.context.current_state.value} -> {target_state.value}. "
                f"Allowed target states: {[s.value for s in ALLOWED_TRANSITIONS.get(self.context.current_state, [])]}"
            )
            logger.error(err_msg)
            self.context.errors.append(err_msg)
            raise StateTransitionError(err_msg)

        previous_state = self.context.current_state
        record = StateTransitionRecord(
            from_state=previous_state,
            to_state=target_state,
            reason=reason,
            metadata=metadata or {},
        )

        self.context.current_state = target_state
        self.context.history.append({
            "from_state": record.from_state.value,
            "to_state": record.to_state.value,
            "timestamp": record.timestamp,
            "reason": record.reason,
            "metadata": record.metadata,
        })

        logger.info(
            "State transition successful: %s -> %s (Reason: %s)",
            previous_state.value,
            target_state.value,
            reason or "N/A",
        )
        return self.context.current_state

    def reset_for_new_cycle(self) -> None:
        """Resets ephemeral plan, execution data, and chained working memory."""
        self.context.cycle_count += 1
        self.context.active_plan.clear()
        self.context.execution_results.clear()
        self.context.working_memory.clear()
        logger.info("Context reset for new cycle #%d", self.context.cycle_count)

    # ------------------------------------------------------------------
    # Non-Blocking Multi-Task Proposal Ledger
    # ------------------------------------------------------------------

    @property
    def proposals_ledger(self) -> List[Dict[str, Any]]:
        """The persistent global array of human-gated tasks for this agent."""
        ledger = getattr(self.context, "proposals_ledger", None)
        if not isinstance(ledger, list):
            ledger = []
            self.context.proposals_ledger = ledger
        return ledger

    def append_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends a task to the global ledger with a unique ID and a
        `PENDING_APPROVAL` status. Never blocks: the caller continues its plan.
        An entry that already carries a task_id is not duplicated.
        """
        entry = dict(proposal)
        task_id = str(entry.get("task_id") or entry.get("id") or "").strip()
        if not task_id:
            task_id = hashlib_short_id(entry)
        entry["task_id"] = task_id
        entry["status"] = ProposalStatus.PENDING_APPROVAL.value
        # A fresh entry always starts undecided; the human sets this per task.
        entry.setdefault("human_decision", None)

        for existing in self.proposals_ledger:
            if isinstance(existing, dict) and existing.get("task_id") == task_id:
                logger.info("Proposal %s already present in ledger; keeping existing entry.", task_id)
                return existing

        self.proposals_ledger.append(entry)
        logger.info(
            "Ledger append: task %s (%s) queued as PENDING_APPROVAL. Ledger depth now %d.",
            task_id, entry.get("type", "Task"), len(self.proposals_ledger),
        )
        return entry

    def get_proposal(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Looks up a single ledger entry by its unique task ID."""
        for entry in self.proposals_ledger:
            if isinstance(entry, dict) and entry.get("task_id") == task_id:
                return entry
        return None

    def remove_proposal(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Purges a single ledger entry (used when a task is DENIED or COMPLETED)."""
        for index, entry in enumerate(self.proposals_ledger):
            if isinstance(entry, dict) and entry.get("task_id") == task_id:
                removed = self.proposals_ledger.pop(index)
                logger.info("Ledger purge: task %s removed from active array.", task_id)
                return removed
        return None

    def purge_proposals(self, status: str) -> List[Dict[str, Any]]:
        """Removes every ledger entry matching a status and returns the purged entries."""
        target = str(status).strip().upper()
        purged = [e for e in self.proposals_ledger if str(e.get("status")).upper() == target]
        if purged:
            self.context.proposals_ledger = [
                e for e in self.proposals_ledger if str(e.get("status")).upper() != target
            ]
            logger.info("Ledger purge: removed %d entr(ies) with status %s.", len(purged), target)
        return purged

    def pending_proposals(self) -> List[Dict[str, Any]]:
        """Returns every ledger entry still awaiting a human decision."""
        return [
            e for e in self.proposals_ledger
            if str(e.get("status", "")).strip().upper() == ProposalStatus.PENDING_APPROVAL.value
        ]

    def ledger_summary(self) -> Dict[str, int]:
        """Compact status histogram used in prompts, logs, and the dashboard."""
        summary: Dict[str, int] = {}
        for entry in self.proposals_ledger:
            key = str(entry.get("status", "UNKNOWN")).strip().upper() or "UNKNOWN"
            summary[key] = summary.get(key, 0) + 1
        return summary

    # ------------------------------------------------------------------
    # Venture / Portfolio Ledger
    # ------------------------------------------------------------------
    # The ventures_ledger is the agent's commercial memory: every product,
    # program, listing, or micro-business it launches is registered here and
    # tracked across stateless runs. Unlike proposals_ledger it stores money
    # facts (revenue, costs, units sold, visitors) and derives a health signal
    # so the agent can tell winners from losers without a human in the loop.

    def ventures(self) -> List[Dict[str, Any]]:
        """The live ventures_ledger array (never None)."""
        ledger = self.context.ventures_ledger
        return ledger if isinstance(ledger, list) else []

    @staticmethod
    def compute_venture_health(venture: Dict[str, Any]) -> str:
        """
        Derives a commercial health signal for a venture from its metrics.

        Rules (first match wins):
          - RETIRED ventures are always RETIRED.
          - No revenue recorded AND zero units sold => NO_SALES (nobody is buying).
          - Revenue > costs  => PROFITABLE.
          - Revenue == costs and revenue > 0 => BREAKEVEN.
          - Revenue < costs  => LOSING (burning money).
          - Otherwise (no financial data at all) => UNKNOWN.
        """
        if str(venture.get("status", "")).upper() == VentureStatus.RETIRED.value:
            return VentureStatus.RETIRED.value

        try:
            revenue = float(venture.get("revenue", 0) or 0)
        except (TypeError, ValueError):
            revenue = 0.0
        try:
            costs = float(venture.get("costs", 0) or 0)
        except (TypeError, ValueError):
            costs = 0.0
        try:
            units = int(venture.get("units_sold", 0) or 0)
        except (TypeError, ValueError):
            units = 0

        if revenue == 0.0 and units <= 0:
            # Either genuinely no sales, or never had financial data recorded.
            if "revenue" not in venture and "units_sold" not in venture:
                return "UNKNOWN"
            return VentureStatus.NO_SALES.value

        profit = revenue - costs
        if profit > 0:
            return VentureStatus.PROFITABLE.value
        if profit == 0:
            return VentureStatus.BREAKEVEN.value
        return VentureStatus.LOSING.value

    @staticmethod
    def _recompute_health(venture: Dict[str, Any]) -> Dict[str, Any]:
        """Stamps profit + health onto a venture dict in place (and returns it)."""
        try:
            revenue = float(venture.get("revenue", 0) or 0)
        except (TypeError, ValueError):
            revenue = 0.0
        try:
            costs = float(venture.get("costs", 0) or 0)
        except (TypeError, ValueError):
            costs = 0.0
        venture["revenue"] = revenue
        venture["costs"] = costs
        venture["profit"] = round(revenue - costs, 4)
        venture["health"] = StateManager.compute_venture_health(venture)
        venture["last_checked"] = datetime.now(timezone.utc).isoformat()
        return venture

    def register_venture(self, venture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registers a new venture (product/program/listing) in the portfolio ledger.

        A unique `venture_id` is assigned if absent, health/profit are computed,
        and the entry is appended. Re-registering the same `venture_id` returns
        the existing entry unchanged (mirrors append_proposal dedup semantics).
        """
        entry = dict(venture)
        venture_id = str(entry.get("venture_id") or entry.get("id") or "").strip()
        if not venture_id:
            venture_id = hashlib_short_id(entry)
        entry["venture_id"] = venture_id
        entry.setdefault("status", VentureStatus.LIVE.value)
        entry.setdefault("revenue", 0.0)
        entry.setdefault("costs", 0.0)
        entry.setdefault("units_sold", 0)
        entry.setdefault("visitors", 0)
        entry.setdefault("created_at", datetime.now(timezone.utc).isoformat())

        for existing in self.ventures():
            if isinstance(existing, dict) and existing.get("venture_id") == venture_id:
                logger.info("Venture %s already present in ledger; keeping existing entry.", venture_id)
                return existing

        self._recompute_health(entry)
        self.ventures().append(entry)
        logger.info(
            "Venture register: %s (%s) added. Portfolio depth now %d.",
            venture_id, entry.get("name", "Unnamed"), len(self.ventures()),
        )
        return entry

    def get_venture(self, venture_id: str) -> Optional[Dict[str, Any]]:
        """Looks up a single venture by its unique ID."""
        for entry in self.ventures():
            if isinstance(entry, dict) and entry.get("venture_id") == venture_id:
                return entry
        return None

    def update_venture(self, venture_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Applies a partial update to a venture and recomputes profit + health.

        Unknown IDs return None. The `venture_id` field itself is immutable.
        """
        entry = self.get_venture(venture_id)
        if entry is None:
            logger.warning("Venture update target %s not found in ledger.", venture_id)
            return None
        for key, value in updates.items():
            if key in ("venture_id", "id"):
                continue
            entry[key] = value
        self._recompute_health(entry)
        logger.info(
            "Venture update: %s -> status=%s health=%s profit=%s",
            venture_id, entry.get("status"), entry.get("health"), entry.get("profit"),
        )
        return entry

    def record_venture_metric(
        self,
        venture_id: str,
        revenue: Optional[float] = None,
        costs: Optional[float] = None,
        units_sold: Optional[int] = None,
        visitors: Optional[int] = None,
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Records a point-in-time commercial metric snapshot for a venture.

        Numeric fields are treated as cumulative-to-date values (the latest
        snapshot wins). A bounded history of snapshots is kept so trends
        (rising/falling sales) remain visible across runs. Health is recomputed.
        """
        entry = self.get_venture(venture_id)
        if entry is None:
            logger.warning("Venture metric target %s not found in ledger.", venture_id)
            return None

        if revenue is not None:
            try:
                entry["revenue"] = float(revenue)
            except (TypeError, ValueError):
                pass
        if costs is not None:
            try:
                entry["costs"] = float(costs)
            except (TypeError, ValueError):
                pass
        if units_sold is not None:
            try:
                entry["units_sold"] = int(units_sold)
            except (TypeError, ValueError):
                pass
        if visitors is not None:
            try:
                entry["visitors"] = int(visitors)
            except (TypeError, ValueError):
                pass

        snapshot = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "revenue": entry.get("revenue", 0.0),
            "costs": entry.get("costs", 0.0),
            "units_sold": entry.get("units_sold", 0),
            "visitors": entry.get("visitors", 0),
        }
        if note:
            snapshot["note"] = str(note)

        history = entry.get("history")
        if not isinstance(history, list):
            history = []
        history.append(snapshot)
        # Keep history bounded so the Gist state file stays small.
        entry["history"] = history[-20:]

        self._recompute_health(entry)
        logger.info(
            "Venture metric: %s revenue=%s costs=%s units=%s health=%s",
            venture_id, entry.get("revenue"), entry.get("costs"),
            entry.get("units_sold"), entry.get("health"),
        )
        return entry

    def portfolio_summary(self) -> Dict[str, Any]:
        """
        Aggregates the whole portfolio into a single commercial picture.

        Returns total revenue/costs/profit, counts per health bucket, and the
        best/worst performers. Used to brief the agent each PLAN/EVALUATE phase
        so it can double down on winners and fix or kill losers.
        """
        ventures = [v for v in self.ventures() if isinstance(v, dict)]
        total_revenue = 0.0
        total_costs = 0.0
        total_units = 0
        total_visitors = 0
        health_counts: Dict[str, int] = {}
        for v in ventures:
            try:
                total_revenue += float(v.get("revenue", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                total_costs += float(v.get("costs", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                total_units += int(v.get("units_sold", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                total_visitors += int(v.get("visitors", 0) or 0)
            except (TypeError, ValueError):
                pass
            health = str(v.get("health", "UNKNOWN"))
            health_counts[health] = health_counts.get(health, 0) + 1

        def _profit_of(v: Dict[str, Any]) -> float:
            try:
                return float(v.get("profit", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        best = max(ventures, key=_profit_of, default=None)
        worst = min(ventures, key=_profit_of, default=None)

        return {
            "total_ventures": len(ventures),
            "total_revenue": round(total_revenue, 4),
            "total_costs": round(total_costs, 4),
            "total_profit": round(total_revenue - total_costs, 4),
            "total_units_sold": total_units,
            "total_visitors": total_visitors,
            "health_counts": health_counts,
            "best_venture": best.get("venture_id") if best else None,
            "worst_venture": worst.get("venture_id") if worst else None,
        }


def hashlib_short_id(entry: Dict[str, Any]) -> str:
    """
    Derives a stable 12-char identifier for a ledger entry that arrives without one.
    Uses only the standard library so the FSM layer stays dependency-free.
    """
    import hashlib
    import json as _json

    try:
        blob = _json.dumps(entry, sort_keys=True, default=str)
    except Exception:
        blob = str(entry)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
