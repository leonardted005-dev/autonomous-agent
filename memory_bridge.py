"""
Memory Bridge Module
====================
Provides an abstract structural interface and local reference implementation
to bridge agent state and execution memories to external storage layers.

In stateless cloud environments (e.g., GitHub Actions runners), the filesystem
is wiped after every run. This module ensures short-term state and long-term logs/knowledge
can be seamlessly persisted and rehydrated across stateless worker invocations.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MemoryBridge")


@dataclass
class ShortTermState:
    """Represents mutable agent state between immediate consecutive execution steps/cycles."""
    run_id: str
    current_state: str
    cycle_count: int = 0
    active_plan: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    session_cookies: Dict[str, str] = field(default_factory=dict)
    proposals_ledger: List[Dict[str, Any]] = field(default_factory=list)
    ventures_ledger: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes to a plain JSON-compatible dictionary.

        `proposals_ledger`, `ventures_ledger`, and `session_cookies` are always emitted
        (never omitted when empty) so downstream readers and cloud state migrations can
        rely on the keys being structurally present regardless of how this instance was
        constructed.
        """
        return {
            "run_id": self.run_id,
            "current_state": self.current_state,
            "cycle_count": self.cycle_count,
            "active_plan": self.active_plan if isinstance(self.active_plan, list) else [],
            "context_data": self.context_data if isinstance(self.context_data, dict) else {},
            "session_cookies": self.session_cookies if isinstance(self.session_cookies, dict) else {},
            "proposals_ledger": self._normalize_ledger(self.proposals_ledger),
            "ventures_ledger": self._normalize_ledger(self.ventures_ledger),
            "last_updated": self.last_updated,
        }

    @staticmethod
    def _normalize_ledger(raw_ledger: Any) -> List[Dict[str, Any]]:
        """
        Coerces any incoming ledger payload into a clean list of dictionaries.
        Non-list payloads degrade to [] and non-dict entries are dropped, so a
        malformed or partially-migrated state file can never propagate bad types.
        """
        if not isinstance(raw_ledger, list):
            return []
        return [dict(entry) for entry in raw_ledger if isinstance(entry, dict)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShortTermState:
        """
        Hydrates an instance from a dictionary, tolerating missing keys and partially
        migrated legacy payloads. Every structural attribute is read defensively and
        falls back to a safe default rather than raising.
        """
        if not isinstance(data, dict):
            data = {}

        raw_cookies = data.get("session_cookies", {})
        cookies = {str(k): str(v) for k, v in raw_cookies.items()} if isinstance(raw_cookies, dict) else {}

        # Prefer the top-level ledger; fall back to a ledger mirrored inside context_data.
        raw_context = data.get("context_data", {})
        context_data = dict(raw_context) if isinstance(raw_context, dict) else {}
        raw_ledger = data.get("proposals_ledger", None)
        if raw_ledger is None:
            raw_ledger = context_data.get("proposals_ledger", [])

        # Same defensive routing for the venture/portfolio ledger.
        raw_ventures = data.get("ventures_ledger", None)
        if raw_ventures is None:
            raw_ventures = context_data.get("ventures_ledger", [])

        raw_plan = data.get("active_plan", [])

        return cls(
            run_id=data.get("run_id", "default_run"),
            current_state=data.get("current_state", "WAKE_UP"),
            cycle_count=data.get("cycle_count", 0) or 0,
            active_plan=list(raw_plan) if isinstance(raw_plan, list) else [],
            context_data=context_data,
            session_cookies=cookies,
            proposals_ledger=cls._normalize_ledger(raw_ledger),
            ventures_ledger=cls._normalize_ledger(raw_ventures),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class MemoryLogEntry:
    """Individual entry in the long-term append-only ledger."""
    entry_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = "INFO"  # INFO, PLAN, ACTION, EVALUATION, ERROR
    source: str = "agent_core"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryLogEntry:
        return cls(
            entry_id=data.get("entry_id", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            event_type=data.get("event_type", "INFO"),
            source=data.get("source", "agent_core"),
            payload=data.get("payload", {}),
        )


class BaseMemoryBridge(ABC):
    """
    Abstract interface for state persistence across stateless cloud runs.
    Subclasses can implement Supabase, DynamoDB, Redis, S3, or GitHub Artifact storage.
    """

    @abstractmethod
    def load_short_term_state(self, agent_id: str) -> Optional[ShortTermState]:
        """Loads the most recent short-term state for a given agent identifier."""
        raise NotImplementedError

    @abstractmethod
    def save_short_term_state(self, agent_id: str, state: ShortTermState) -> bool:
        """Saves short-term state snapshot for a given agent identifier."""
        raise NotImplementedError

    @abstractmethod
    def append_long_term_log(self, agent_id: str, entry: MemoryLogEntry) -> bool:
        """Appends an event/log entry into persistent append-only storage."""
        raise NotImplementedError

    @abstractmethod
    def retrieve_recent_logs(self, agent_id: str, limit: int = 50) -> List[MemoryLogEntry]:
        """Retrieves recent long-term memory logs for agent context enrichment."""
        raise NotImplementedError

    @abstractmethod
    def clear_short_term_state(self, agent_id: str) -> bool:
        """Clears short-term state (e.g. upon graceful run shutdown or reset)."""
        raise NotImplementedError


class LocalFileMemoryBridge(BaseMemoryBridge):
    """
    Local filesystem reference implementation of BaseMemoryBridge.
    Stores short-term snapshots in JSON files and long-term logs in JSONL files.
    Ideal for local testing, development, or mounted persistent volumes.
    """

    def __init__(self, storage_dir: str = ".agent_memory") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.storage_dir / "state"
        self.logs_dir = self.storage_dir / "logs"
        self.state_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        logger.info("LocalFileMemoryBridge initialized at %s", self.storage_dir.resolve())

    def _state_file_path(self, agent_id: str) -> Path:
        sanitized = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
        return self.state_dir / f"{sanitized}_state.json"

    def _log_file_path(self, agent_id: str) -> Path:
        sanitized = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
        return self.logs_dir / f"{sanitized}_logs.jsonl"

    def load_short_term_state(self, agent_id: str) -> Optional[ShortTermState]:
        path = self._state_file_path(agent_id)
        if not path.exists():
            logger.info("No prior short-term state found for agent '%s'", agent_id)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Successfully loaded short-term state for '%s'", agent_id)
            return ShortTermState.from_dict(data)
        except Exception as e:
            logger.error("Failed to load state for '%s': %s", agent_id, e)
            return None

    def save_short_term_state(self, agent_id: str, state: ShortTermState) -> bool:
        path = self._state_file_path(agent_id)
        try:
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            temp_path.replace(path)
            logger.info("Saved short-term state for '%s' (State: %s)", agent_id, state.current_state)
            return True
        except Exception as e:
            logger.error("Failed to save state for '%s': %s", agent_id, e)
            return False

    def append_long_term_log(self, agent_id: str, entry: MemoryLogEntry) -> bool:
        path = self._log_file_path(agent_id)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error("Failed to append log for '%s': %s", agent_id, e)
            return False

    def retrieve_recent_logs(self, agent_id: str, limit: int = 50) -> List[MemoryLogEntry]:
        path = self._log_file_path(agent_id)
        if not path.exists():
            return []
        entries: List[MemoryLogEntry] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(MemoryLogEntry.from_dict(json.loads(line)))
            return entries[-limit:]
        except Exception as e:
            logger.error("Failed to read logs for '%s': %s", agent_id, e)
            return []

    def clear_short_term_state(self, agent_id: str) -> bool:
        path = self._state_file_path(agent_id)
        if path.exists():
            try:
                path.unlink()
                logger.info("Cleared short-term state for '%s'", agent_id)
                return True
            except Exception as e:
                logger.error("Failed to clear short-term state for '%s': %s", agent_id, e)
                return False
        return True

