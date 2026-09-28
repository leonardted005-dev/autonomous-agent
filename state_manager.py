"""
State Manager Module
====================
Provides an explicit Finite State Machine (FSM) tailored for stateless cloud runner environments
such as GitHub Actions, AWS Lambda, or cron container workers.

The agent transitions sequentially through lifecycle states:
    WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP (or ERROR / PENDING_APPROVAL)

Each state change is validated against allowed transitions to prevent invalid execution paths
and ensure complete determinism across ephemeral worker instances.
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
ALLOWED_TRANSITIONS: Dict[AgentState, List[AgentState]] = {
    AgentState.WAKE_UP: [AgentState.PLAN, AgentState.ERROR, AgentState.SLEEP],
    AgentState.PLAN: [AgentState.EXECUTE, AgentState.EVALUATE, AgentState.PENDING_APPROVAL, AgentState.ERROR, AgentState.SLEEP],
    AgentState.EXECUTE: [AgentState.EVALUATE, AgentState.PLAN, AgentState.PENDING_APPROVAL, AgentState.ERROR, AgentState.SLEEP],
    AgentState.EVALUATE: [AgentState.PLAN, AgentState.SLEEP, AgentState.ERROR],
    AgentState.PENDING_APPROVAL: [AgentState.PLAN, AgentState.EXECUTE, AgentState.SLEEP, AgentState.ERROR, AgentState.WAKE_UP],
    AgentState.SLEEP: [AgentState.WAKE_UP],
    AgentState.ERROR: [AgentState.SLEEP],  # Graceful wind-down after failure logging
}


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
    errors: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes context to a plain JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "cycle_count": self.cycle_count,
            "current_state": self.current_state.value,
            "payload": self.payload,
            "active_plan": self.active_plan,
            "execution_results": self.execution_results,
            "errors": self.errors,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentContext:
        """Hydrates context instance from a dictionary."""
        raw_state = data.get("current_state", AgentState.WAKE_UP.value)
        try:
            current_state = AgentState(raw_state)
        except ValueError:
            current_state = AgentState.WAKE_UP

        return cls(
            run_id=data.get("run_id", "unknown_run"),
            cycle_count=data.get("cycle_count", 0),
            current_state=current_state,
            payload=data.get("payload", {}),
            active_plan=data.get("active_plan", []),
            execution_results=data.get("execution_results", []),
            errors=data.get("errors", []),
            history=data.get("history", []),
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
        """Resets ephemeral plan and execution data while advancing cycle counter."""
        self.context.cycle_count += 1
        self.context.active_plan.clear()
        self.context.execution_results.clear()
        logger.info("Context reset for new cycle #%d", self.context.cycle_count)

    to_state: AgentState
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
