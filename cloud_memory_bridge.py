"""
Cloud Memory Bridge Module
==========================
Stateless cloud persistence layer implementing BaseMemoryBridge.
Leverages the official GitHub Gists API via Python's standard library (urllib.request)
to save and load agent state across ephemeral runners (e.g., GitHub Actions).

Key Capabilities:
1. Zero heavy/paid dependencies - relies solely on Python built-ins.
2. Authenticates via GITHUB_TOKEN environment variable.
3. Manages short-term state in a designated JSON file inside a private Gist.
4. Manages long-term audit logs in a designated JSONL file inside the Gist.
5. Graceful fallback: If the Gist API fails, network drops, or credentials are invalid,
   it transparently falls back to LocalFileMemoryBridge or raises structured architectural warnings.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from memory_bridge import BaseMemoryBridge, LocalFileMemoryBridge, MemoryLogEntry, ShortTermState

logger = logging.getLogger("CloudMemoryBridge")


class GitHubGistMemoryBridge(BaseMemoryBridge):
    """
    Stateless memory bridge storing agent state and append-only audit logs
    in a private GitHub Gist using the official GitHub REST API.
    """

    DEFAULT_API_BASE: str = "https://api.github.com"

    def __init__(
        self,
        gist_id: Optional[str] = None,
        github_token: Optional[str] = None,
        api_base: Optional[str] = None,
        fallback_bridge: Optional[BaseMemoryBridge] = None,
        timeout: int = 20,
    ) -> None:
        self.gist_id = gist_id or os.getenv("GIST_ID") or os.getenv("MEMORY_GIST_ID")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.api_base = (api_base or os.getenv("GITHUB_API_BASE") or self.DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self.fallback = fallback_bridge or LocalFileMemoryBridge()

        if not self.github_token:
            logger.warning(
                "GITHUB_TOKEN is not configured. GitHubGistMemoryBridge will route requests to fallback."
            )
        if not self.gist_id:
            logger.warning(
                "GIST_ID is not configured. GitHubGistMemoryBridge will route requests to fallback."
            )

    @property
    def is_configured(self) -> bool:
        """Checks if both GIST_ID and GITHUB_TOKEN are set."""
        return bool(self.gist_id and self.github_token)

    def _state_filename(self, agent_id: str) -> str:
        sanitized = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
        return f"{sanitized}_state.json"

    def _log_filename(self, agent_id: str) -> str:
        sanitized = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
        return f"{sanitized}_logs.jsonl"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "AutonomousAgent-MemoryBridge/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    def _fetch_gist_data(self) -> Optional[Dict[str, Any]]:
        """Performs a GET request to retrieve Gist payload."""
        if not self.is_configured:
            return None

        url = f"{self.api_base}/gists/{self.gist_id}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
                logger.error("GitHub Gist API responded with status %d", resp.status)
                return None
        except urllib.error.HTTPError as http_err:
            body = http_err.read().decode("utf-8", errors="replace")
            logger.error("HTTP error reading Gist (%d): %s", http_err.code, body)
            return None
        except Exception as e:
            logger.error("Network/unexpected error reading Gist: %s", e)
            return None

    def _patch_gist_files(self, files_payload: Dict[str, Dict[str, Optional[str]]]) -> bool:
        """Performs a PATCH request to update file(s) in the private Gist."""
        if not self.is_configured:
            return False

        url = f"{self.api_base}/gists/{self.gist_id}"
        body = json.dumps({"files": files_payload}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._get_headers(), method="PATCH")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 201):
                    return True
                logger.error("GitHub Gist API update responded with status %d", resp.status)
                return False
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            logger.error("HTTP error updating Gist (%d): %s", http_err.code, err_body)
            return False
        except Exception as e:
            logger.error("Network/unexpected error updating Gist: %s", e)
            return False


    def load_short_term_state(self, agent_id: str) -> Optional[ShortTermState]:
        """Loads state from the private Gist, falling back to local storage on error."""
        if not self.is_configured:
            logger.info("Gist bridge not configured. Falling back to local storage for load_short_term_state.")
            return self.fallback.load_short_term_state(agent_id)

        filename = self._state_filename(agent_id)
        gist_data = self._fetch_gist_data()

        if not gist_data:
            logger.warning("Falling back to local storage for load_short_term_state due to Gist fetch failure.")
            return self.fallback.load_short_term_state(agent_id)

        files = gist_data.get("files", {})
        file_obj = files.get(filename)
        if not file_obj or "content" not in file_obj:
            logger.info("File '%s' not present in Gist '%s'. Falling back to local storage.", filename, self.gist_id)
            return self.fallback.load_short_term_state(agent_id)

        try:
            content = file_obj["content"]
            parsed = json.loads(content)
            logger.info("Successfully loaded short-term state for '%s' from Gist '%s'", agent_id, self.gist_id)
            return ShortTermState.from_dict(parsed)
        except Exception as e:
            logger.error("Failed to parse state JSON from Gist file '%s': %s", filename, e)
            return self.fallback.load_short_term_state(agent_id)

    def save_short_term_state(self, agent_id: str, state: ShortTermState) -> bool:
        """Saves state into the private Gist, mirroring to local fallback on failure or success."""
        # Mirror to local fallback first for runner-level persistence
        self.fallback.save_short_term_state(agent_id, state)

        if not self.is_configured:
            logger.info("Gist bridge not configured. Short-term state saved only to local fallback.")
            return True

        filename = self._state_filename(agent_id)
        payload = json.dumps(state.to_dict(), indent=2)
        success = self._patch_gist_files({filename: {"content": payload}})

        if success:
            logger.info("Successfully saved short-term state for '%s' to Gist '%s'", agent_id, self.gist_id)
            return True
        else:
            logger.warning("Failed to save state to Gist. Local fallback retained.")
            return False


    def append_long_term_log(self, agent_id: str, entry: MemoryLogEntry) -> bool:
        """Appends an audit log into the Gist log file, mirroring to local fallback."""
        self.fallback.append_long_term_log(agent_id, entry)

        if not self.is_configured:
            return True

        filename = self._log_filename(agent_id)
        current_content = ""
        gist_data = self._fetch_gist_data()
        if gist_data and "files" in gist_data and filename in gist_data["files"]:
            current_content = gist_data["files"][filename].get("content", "") or ""

        entry_line = json.dumps(entry.to_dict()) + "\n"
        updated_content = current_content + entry_line
        return self._patch_gist_files({filename: {"content": updated_content}})

    def retrieve_recent_logs(self, agent_id: str, limit: int = 50) -> List[MemoryLogEntry]:
        """Retrieves recent logs from Gist, falling back to local storage."""
        if not self.is_configured:
            return self.fallback.retrieve_recent_logs(agent_id, limit)

        filename = self._log_filename(agent_id)
        gist_data = self._fetch_gist_data()
        if not gist_data or "files" not in gist_data or filename not in gist_data["files"]:
            return self.fallback.retrieve_recent_logs(agent_id, limit)

        content = gist_data["files"][filename].get("content", "")
        entries: List[MemoryLogEntry] = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(MemoryLogEntry.from_dict(json.loads(line)))
                except Exception:
                    continue
        return entries[-limit:] if entries else self.fallback.retrieve_recent_logs(agent_id, limit)

    def clear_short_term_state(self, agent_id: str) -> bool:
        """Clears short-term state in both local fallback and Gist."""
        self.fallback.clear_short_term_state(agent_id)
        if not self.is_configured:
            return True
        filename = self._state_filename(agent_id)
        # In GitHub Gist API, passing null removes the file
        return self._patch_gist_files({filename: None})  # type: ignore[dict-item]

