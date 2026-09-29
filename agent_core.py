"""
Agent Core Module
=================
Main autonomous execution loop for running on stateless cloud infrastructure
(such as scheduled GitHub Actions workflows or serverless functions).

Core Responsibilities:
1. Orchestrates the Finite State Machine (WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP).
2. Hydrates and persists state via MemoryBridge.
3. Invokes the AI Brain via standard environment variables (OPENAI_API_KEY, GROQ_API_KEY,
   or ANTHROPIC_API_KEY) with pure Python standard libraries (urllib.request).
4. Enforces an internal structured monologue ("Thinking Process", "Step-by-Step Plan",
   "Execution Assessment").
5. Implements resilient error handling and boundary safety to prevent unhandled crashes
   and ensure graceful shutdown / state preservation under failure.
6. Maintains a Non-Blocking Multi-Task Proposal Ledger: human-gated work is appended to a
   persistent global array (`proposals_ledger`) with a unique ID and a `PENDING_APPROVAL`
   status, and the agent immediately proceeds to its next planned pipeline step within the
   same run. The lifecycle only settles into SLEEP once the whole multi-step plan for the
   cycle has concluded, so many verifications can wait on the dashboard in parallel while
   the agent keeps exploring new opportunities.
7. Injects a human-provided `session_cookies` map from the Gist state into outgoing
   request headers as standard Cookie literals, so the agent can work inside an
   environment the operator logged into from the dashboard without ever seeing a password.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import os
import re
import smtplib
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_bridge import BaseMemoryBridge, LocalFileMemoryBridge, MemoryLogEntry, ShortTermState
from cloud_memory_bridge import GitHubGistMemoryBridge
from state_manager import (
    AgentContext,
    AgentState,
    ProposalStatus,
    StateManager,
)

# Maximum sequential execution steps (internal tool-chain links) allowed per cycle
# before the run concludes and the FSM settles into SLEEP.
MAX_CHAIN_STEPS_PER_CYCLE = 5

# Setup structured logger
logging.basicConfig(
    level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AgentCore")


@dataclass
class AgentConfig:
    """Agent runtime configuration sourced from environment variables."""
    agent_id: str = os.getenv("AGENT_ID", "default_stateless_agent")
    max_steps_per_run: int = int(os.getenv("MAX_STEPS_PER_RUN", "5"))
    max_tool_calls_per_cycle: int = int(os.getenv("MAX_TOOL_CALLS_PER_CYCLE", str(MAX_CHAIN_STEPS_PER_CYCLE)))
    llm_provider: str = os.getenv(
        "LLM_PROVIDER",
        "groq" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
    ).lower()
    llm_model: str = os.getenv(
        "LLM_MODEL",
        "qwen/qwen3.8-27b" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "gpt-4o-mini"
    )
    llm_api_base: str = os.getenv(
        "LLM_API_BASE",
        "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "https://api.openai.com/v1"
    )
    request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "45"))
    github_token: Optional[str] = os.getenv("CUSTOM_GIT_TOKEN")
    gist_id: Optional[str] = os.getenv("GIST_ID") or os.getenv("MEMORY_GIST_ID")


def create_memory_bridge(config: Optional[AgentConfig] = None) -> BaseMemoryBridge:
    """
    Factory function resolving appropriate memory bridge based on runtime environment.
    Automatically switches to GitHubGistMemoryBridge if CUSTOM_GIT_TOKEN and GIST_ID are present,
    otherwise uses LocalFileMemoryBridge with local disk persistence.
    """
    cfg = config or AgentConfig()
    token = cfg.github_token
    gist_id = cfg.gist_id

    if token and gist_id:
        logger.info(
            "Cloud environment detected (CUSTOM_GIT_TOKEN and GIST_ID found). "
            "Initializing GitHubGistMemoryBridge (Gist: %s).",
            gist_id,
        )
        return GitHubGistMemoryBridge(gist_id=gist_id, github_token=token)
    elif token or gist_id:
        missing = "GIST_ID" if not gist_id else "CUSTOM_GIT_TOKEN"
        logger.warning(
            "Partial cloud memory credentials detected (%s missing). "
            "Defaulting to LocalFileMemoryBridge.",
            missing,
        )

    logger.info("Using LocalFileMemoryBridge for memory persistence.")
    return LocalFileMemoryBridge()

class Toolbox:
    """
    Zero-dependency autonomous capabilities engine.
    Equips the agent with standard library tools for environment perception,
    local asset authoring, and external communication.
    """

    def __init__(self, workspace_root: Optional[str] = None, request_timeout: int = 15) -> None:
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.request_timeout = request_timeout
        logger.info("Toolbox initialized with workspace root: %s", self.workspace_root)

    @staticmethod
    def _format_cookie_header(session_cookies: Optional[Dict[str, str]]) -> str:
        """
        Formats a cookie dict ('name' -> 'value') into a standard Cookie header string.
        Entries with empty names are dropped; control characters are stripped for safety.
        """
        if not session_cookies or not isinstance(session_cookies, dict):
            return ""
        pairs = []
        for name, value in session_cookies.items():
            clean_name = re.sub(r"[\x00-\x1F\x7F;\s=]", "", str(name))
            clean_value = re.sub(r"[\x00-\x1F\x7F;]", "", str(value)).strip()
            if clean_name and clean_value:
                pairs.append(f"{clean_name}={clean_value}")
        return "; ".join(pairs)

    @staticmethod
    def parse_cookie_string(cookie_string: str) -> Dict[str, str]:
        """
        Parses a raw browser cookie string ('name=value; name2=value2') pasted from
        developer tools into a clean name -> value dict raedy for session injection.
        Attributes without '=' (Secure, HttpOnly, SameSite, Path...) are skipped.
        """
        cookies: Dict[str, str] = {}
        if not cookie_string or not isinstance(cookie_string, str):
            return cookies
        for chunk in cookie_string.split(";"):
            chunk = chunk.strip()
            if "=" not in chunk:
                continue
            name, _, value = chunk.partition("=")
            name = name.strip()
            value = value.strip().strip('"')
            if name and value:
                cookies[name] = value
        return cookies

    def web_browser_tool(self, url: str, session_cookies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetches text/HTML/RSS content from web endpoints using standard urllib.request.
        Strips scripts, styles, and markup to return clean textual data.
        If a 'session_cookies' dict is provided (e.g. from Gist state), it is injected
        as a standard 'Cookie' header so authenticated pages can be read.
        """
        logger.info("Toolbox: web_browser_tool called on URL: %s", url)
        if not url or not isinstance(url, str):
            return {"status": "ERROR", "error": "Invalid URL provided: URL must be a non-empty string."}

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"status": "ERROR", "error": f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."}

        cookie_header = self._format_cookie_header(session_cookies)
        request_headers = {
            "User-Agent": "AutonomousAgent-Toolbox/1.0 (+https://github.com/leonardted005-dev/autonomous-agent)",
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,application/json,*/*",
        }
        if cookie_header:
            request_headers["Cookie"] = cookie_header
            logger.info("Toolbox: injecting %d session cookie(s) into web request.", len(session_cookies or {}))

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read()
                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

                raw_text = raw_bytes.decode(encoding, errors="replace")
                clean_text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
                clean_text = re.sub(r"<script.*?</script>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
                clean_text = re.sub(r"<style.*?</style>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
                clean_text = re.sub(r"<[^>]+>", " ", clean_text)
                unescaped = html_lib.unescape(clean_text)
                normalized = " ".join(unescaped.split())
                preview = normalized[:4000]

                return {
                    "status": "SUCCESS",
                    "url": url,
                    "http_status": resp.status,
                    "content_length": len(normalized),
                    "content_preview": preview,
                }
        except urllib.error.HTTPError as http_err:
            msg = f"HTTP {http_err.code} while fetching {url}: {http_err.reason}"
            logger.warning("web_browser_tool error: %s", msg)
            return {"status": "ERROR", "url": url, "http_status": http_err.code, "error": msg}
        except Exception as e:
            msg = f"Failed to access {url}: {str(e)}"
            logger.warning("web_browser_tool error: %s", msg)
            return {"status": "ERROR", "url": url, "error": msg}

    def file_writer_tool(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Safely writes out scripts, reports, or data files into the workspace directory.
        Enforces path containment within the workspace root.
        """
        logger.info("Toolbox: file_writer_tool called for filepath: %s", filepath)
        if not filepath or not isinstance(filepath, str):
            return {"status": "ERROR", "error": "Invalid filepath provided."}

        try:
            target_path = (self.workspace_root / filepath).resolve()
            if not str(target_path).startswith(str(self.workspace_root)):
                err = f"Security violation: path '{filepath}' resolves outside workspace root."
                logger.error(err)
                return {"status": "ERROR", "error": err}

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info("Successfully wrote %d bytes to %s", len(content), target_path)
            return {
                "status": "SUCCESS",
                "filepath": str(target_path.relative_to(self.workspace_root)),
                "bytes_written": len(content),
            }
        except Exception as e:
            msg = f"Failed to write file '{filepath}': {str(e)}"
            logger.error("file_writer_tool error: %s", msg)
            return {"status": "ERROR", "filepath": filepath, "error": msg}

    def http_poster_tool(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Sends HTTP POST requests with JSON payload to simulate platform interactions,
        job submissions, webhook dispatches, or remote logging.
        """
        logger.info("Toolbox: http_poster_tool called on URL: %s", url)
        if not url or not isinstance(url, str):
            return {"status": "ERROR", "error": "Invalid URL provided: URL must be a non-empty string."}

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"status": "ERROR", "error": f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."}

        req_headers = {
            "Content-Type": "application/json",
            "User-Agent": "AutonomousAgent-Toolbox/1.0 (+https://github.com/leonardted005-dev/autonomous-agent)",
            "Accept": "application/json, text/plain, */*",
        }
        if headers and isinstance(headers, dict):
            req_headers.update(headers)

        body_bytes = json.dumps(payload if payload is not None else {}).encode("utf-8")
        req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                resp_data = resp.read().decode("utf-8", errors="replace")
                parsed_resp: Any = None
                try:
                    parsed_resp = json.loads(resp_data)
                except Exception:
                    parsed_resp = resp_data[:1000]

                return {
                    "status": "SUCCESS",
                    "url": url,
                    "http_status": resp.status,
                    "response": parsed_resp,
                }
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            msg = f"HTTP {http_err.code} on POST {url}: {http_err.reason} - {err_body[:300]}"
            logger.warning("http_poster_tool error: %s", msg)
            return {"status": "ERROR", "url": url, "http_status": http_err.code, "error": msg}
        except Exception as e:
            msg = f"POST request to {url} failed: {str(e)}"
            logger.warning("http_poster_tool error: %s", msg)
            return {"status": "ERROR", "url": url, "error": msg}

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches dynamic tool execution by name with arguments."""
        if tool_name == "web_browser_tool":
            return self.web_browser_tool(url=arguments.get("url", ""), session_cookies=arguments.get("session_cookies"))
        elif tool_name == "file_writer_tool":
            return self.file_writer_tool(
                filepath=arguments.get("filepath", ""),
                content=arguments.get("content", ""),
            )
        elif tool_name == "http_poster_tool":
            return self.http_poster_tool(
                url=arguments.get("url", ""),
                payload=arguments.get("payload", {}),
                headers=arguments.get("headers"),
            )
        else:
            err = f"Unknown tool requested: '{tool_name}'."
            logger.warning(err)
            return {"status": "ERROR", "error": err}


class LLMClient:
    """
    Pure Python standard library (urllib) client for invoking AI model endpoints.
    Eliminates heavy external SDK dependencies to keep cloud runners lightweight.

    Production-grade multi-provider failover chain (Fallback Priority Ledger):
      Priority 1: Groq       (GROQ_API_KEY,       api.groq.com, qwen/qwen3.8-27b)
      Priority 2: Gemini     (GEMINI_API_KEY,     googleapis.com, gemini-1.5-flash)
      Priority 3: Mistral    (MISTRAL_API_KEY,    mistral.ai, mistral-small-latest)
      Priority 4: OpenRouter (OPENROUTER_API_KEY, openrouter.ai, openrouter/free)
      Priority 5: xKiro      (XKIRO_API_KEY,      api.xkiro.com, open-weights fallback)
    """

    # Fallback Priority Ledger: ordered provider chain for rate-limit resilience.
    FALLBACK_PRIORITY_LEDGER: List[Dict[str, str]] = [
        {"name": "groq", "api_base": "https://api.groq.com/openai/v1", "model": "qwen/qwen3.8-27b", "key_env": "GROQ_API_KEY"},
        {"name": "gemini", "api_base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-1.5-flash", "key_env": "GEMINI_API_KEY"},
        {"name": "mistral", "api_base": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key_env": "MISTRAL_API_KEY"},
        {"name": "openrouter", "api_base": "https://openrouter.ai/api/v1", "model": "openrouter/free", "key_env": "OPENROUTER_API_KEY"},
        {"name": "xkiro", "api_base": "https://api.xkiro.com/v1", "model": "qwen/qwen3-32b", "key_env": "XKIRO_API_KEY"},
    ]

    # HTTP statuses that trigger failover to the next priority provider.
    RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.api_key = self._resolve_legacy_api_key()

    def _resolve_legacy_api_key(self) -> Optional[str]:
        if self.config.llm_provider == "groq":
            return os.getenv("GROQ_API_KEY")
        elif self.config.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    def _resolve_api_key(self) -> Optional[str]:
        """Backward-compatible alias for legacy single-key resolution."""
        return self._resolve_legacy_api_key()

    def _resolve_provider_chain(self) -> List[Dict[str, str]]:
        """
        Builds the active failover chain from the Fallback Priority Ledger.
        Providers without configured keys are skipped.
        """
        chain: List[Dict[str, str]] = []
        seen = set()
        preferred = self.config.llm_provider.lower()

        for entry in self.FALLBACK_PRIORITY_LEDGER:
            if entry["name"] == preferred and os.getenv(entry["key_env"]):
                key = str(os.getenv(entry["key_env"])).strip()
                chain.append({"name": entry["name"], "api_base": entry["api_base"], "model": entry["model"], "key": key, "key_env": entry["key_env"]})
                seen.add(entry["name"])
                break

        if preferred in ("openai", "anthropic") and self.api_key:
            chain.append({"name": preferred, "api_base": self.config.llm_api_base, "model": self.config.llm_model, "key": str(self.api_key).strip(), "key_env": "OPENAI_API_KEY"})
            seen.add(preferred)

        for entry in self.FALLBACK_PRIORITY_LEDGER:
            if entry["name"] in seen:
                continue
            key = os.getenv(entry["key_env"])
            if not key:
                continue
            model = os.getenv(entry["name"].upper() + "_MODEL", entry["model"])
            chain.append({"name": entry["name"], "api_base": entry["api_base"], "model": model, "key": str(key).strip(), "key_env": entry["key_env"]})

        if not chain and self.api_key:
            chain.append({"name": preferred, "api_base": self.config.llm_api_base, "model": self.config.llm_model, "key": str(self.api_key).strip(), "key_env": "LLM_API_KEY"})

        return chain

    def _post_chat_completion(self, api_base: str, model: str, api_key: str, messages: List[Dict[str, str]], temperature: float) -> str:
        """Executes a single OpenAI-compatible chat completion POST request."""
        endpoint = f"{api_base.rstrip('/')}/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
            "User-Agent": "StatelessAgentCore/1.0",
            "HTTP-Referer": "https://github.com/leonardted005-dev/autonomous-agent",
            "X-Title": "AutonomousAgent",
        }
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """
        Invokes chat completion across the Fallback Priority Ledger.
        On HTTP 429/400-class errors or network timeouts, diverts the exact
        same prompt state to the next priority engine backup.
        """
        chain = self._resolve_provider_chain()
        if not chain:
            logger.warning(
                "No API key detected for provider '%s'. Simulating heuristic response for offline runner mode.",
                self.config.llm_provider,
            )
            return self._heuristic_fallback(messages)

        last_error: Optional[Exception] = None
        for index, provider in enumerate(chain):
            label = provider["name"] + " (" + provider["model"] + " @ " + provider["api_base"] + ")"
            try:
                logger.info("LLM attempt %d/%d via %s", index + 1, len(chain), label)
                return self._post_chat_completion(provider["api_base"], provider["model"], provider["key"], messages, temperature)
            except urllib.error.HTTPError as http_err:
                error_body = http_err.read().decode("utf-8", errors="replace")
                logger.error("LLM HTTP error %d via %s: %s", http_err.code, label, error_body)
                last_error = RuntimeError("LLM API returned HTTP " + str(http_err.code) + " via " + label + ": " + error_body)
                if http_err.code in self.RETRYABLE_STATUS_CODES and index < len(chain) - 1:
                    print("[WARNING] Active provider failed with error. Diverting prompt state to the next priority engine backup...")
                    logger.warning("Failing over from %s to next priority provider.", label)
                    continue
                raise last_error from http_err
            except Exception as e:
                logger.error("LLM request failed via %s: %s", label, e)
                last_error = e
                if index < len(chain) - 1:
                    print("[WARNING] Active provider failed with error. Diverting prompt state to the next priority engine backup...")
                    logger.warning("Failing over from %s to next priority provider.", label)
                    continue
                raise

        logger.error("All LLM providers exhausted; using offline heuristic. Last error: %s", last_error)
        return self._heuristic_fallback(messages)

    def _heuristic_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Provides predictable structured responses when no API credentials are provided."""
        last_msg = messages[-1]["content"] if messages else ""
        return (
            "THINKING PROCESS:\n"
            "- Operating in autonomous exploration mode without live API key.\n"
            "- Reviewing long-term memory logs and local workspace environment.\n"
            "- Selecting file_writer_tool to record an autonomous discovery log.\n"
            "STEP-BY-STEP PLAN:\n"
            '1. TOOL_CALL: file_writer_tool {"filepath": "autonomous_discovery.txt", "content": "Discovery cycle initialized. Exploring opportunities."}\n'
            "2. Inspect and verify tool output status.\n"
            "3. Transition to evaluation and update long-term memory.\n"
            "EXECUTION ASSESSMENT:\n"
            "Tool execution verified and completed successfully."
        )


class NotificationManager:
    """
    Zero-dependency notification layer (stdlib urllib.request + smtplib).
    Fires instant alerts the moment a proposal enters PENDING_APPROVAL so the
    human reviewer can act before the next lifecycle phase.
    """

    DEFAULT_DASHBOARD_URL = "https://autonomous-agent-gateway.vercel.app"

    def __init__(self, dashboard_url: Optional[str] = None) -> None:
        self.dashboard_url = dashboard_url or os.getenv("DASHBOARD_URL", self.DEFAULT_DASHBOARD_URL)

    # ------------------------------------------------------------------
    # Aggregator
    # ------------------------------------------------------------------
    def send_alerts(self, proposal_json: Dict[str, Any]) -> None:
        """Dispatches all configured alerts. Never raises into the agent loop."""
        self.send_discord_alert(proposal_json)
        self.send_email_alert(proposal_json)

    # ------------------------------------------------------------------
    # 1. Discord Webhook alert
    # ------------------------------------------------------------------
    @staticmethod
    def _field(value: Any, fallback: str = "TBD") -> str:
        """Guarantees a non-empty, size-safe Discord embed field value."""
        text = str(value).strip() if value is not None else ""
        if not text:
            text = fallback
        return text[:1024]

    def send_discord_alert(self, proposal_json: Dict[str, Any]) -> bool:
        """
        Posts an embedded approval card to DISCORD_WEBHOOK_URL with Task ID,
        Platform, Estimated Revenue, and a direct Vercel Dashboard link.
        """
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            logger.debug("DISCORD_WEBHOOK_URL not configured; skipping Discord alert.")
            return False

        task_id = proposal_json.get("task_id", "N/A")
        embed = {
            "title": "New Task Awaiting Your Approval",
            "description": self._field(proposal_json.get("type"), "Task") + " requires human review before execution.",
            "color": 0xFFC107,  # Amber warning
            "fields": [
                {"name": "Task ID", "value": self._field(task_id), "inline": True},
                {"name": "Platform", "value": self._field(proposal_json.get("target_platform")), "inline": True},
                {"name": "Estimated Revenue", "value": self._field(proposal_json.get("estimated_revenue")), "inline": True},
                {"name": "Proposed Action", "value": self._field(proposal_json.get("proposed_action"))[:1024], "inline": False},
                {
                    "name": "Vercel Dashboard",
                    "value": "[Review & Approve](" + self.dashboard_url + ")",
                    "inline": False,
                },
            ],
            "footer": {"text": "Autonomous Agent - Human-in-the-Loop Gateway"},
            "timestamp": proposal_json.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
        payload = {"username": "AutonomousAgent", "content": "Approval required: task " + str(task_id), "embeds": [embed]}
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "StatelessAgentCore/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                ok = getattr(resp, "status", 200) in (200, 204)
            logger.info("Discord alert dispatched for task %s (status ok=%s)", task_id, ok)
            return ok
        except Exception as e:
            logger.warning("Discord alert failed for task %s: %s", task_id, e)
            return False

    # ------------------------------------------------------------------
    # 2. Email alert (SMTP)
    # ------------------------------------------------------------------
    def send_email_alert(self, proposal_json: Dict[str, Any]) -> bool:
        """
        Fires a summary email via SMTP_SERVER/SMTP_PORT/SMTP_USER/SMTP_PASSWORD
        telling the user a task is awaiting their review.
        """
        server = os.getenv("SMTP_SERVER")
        user = os.getenv("SMTP_USER")
        password = os.getenv("SMTP_PASSWORD")
        if not (server and user and password):
            logger.debug("SMTP credentials not configured; skipping email alert.")
            return False

        try:
            port = int(os.getenv("SMTP_PORT", "587"))
        except ValueError:
            logger.warning("Invalid SMTP_PORT '%s'; skipping email alert.", os.getenv("SMTP_PORT"))
            return False

        recipient = os.getenv("NOTIFY_EMAIL", user)
        task_id = proposal_json.get("task_id", "N/A")
        subject = "[Autonomous Agent] Task %s awaiting your approval" % task_id
        body = (
            "A new task proposal has entered PENDING_APPROVAL and needs your review.\n\n"
            "Task ID:            %s\n"
            "Type:               %s\n"
            "Target Platform:    %s\n"
            "Estimated Revenue:  %s\n"
            "Proposed Action:    %s\n\n"
            "Review & decide here: %s\n"
            "Set human_decision to APPROVED or DENIED in the Gist state to proceed.\n"
        ) % (
            task_id,
            proposal_json.get("type", "N/A"),
            proposal_json.get("target_platform", "TBD"),
            proposal_json.get("estimated_revenue", "TBD"),
            str(proposal_json.get("proposed_action", "N/A"))[:500],
            self.dashboard_url,
        )

        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            if port == 465:
                with smtplib.SMTP_SSL(server, port, timeout=15) as smtp:
                    smtp.login(user, password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(server, port, timeout=15) as smtp:
                    smtp.starttls()
                    smtp.login(user, password)
                    smtp.send_message(msg)
            logger.info("Email alert dispatched for task %s to %s", task_id, recipient)
            return True
        except Exception as e:
            logger.warning("Email alert failed for task %s: %s", task_id, e)
            return False


class AutonomousAgent:
    """
    Main Agent Orchestrator. Coordinates FSM execution, monologue generation,
    action dispatch, and memory synchronization across stateless runs.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory_bridge: Optional[BaseMemoryBridge] = None,
        toolbox: Optional[Toolbox] = None,
        notifier: Optional[NotificationManager] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.memory = memory_bridge or create_memory_bridge(self.config)
        self.llm = LLMClient(self.config)
        self.toolbox = toolbox or Toolbox()
        self.notifier = notifier or NotificationManager()
        self.state_manager: Optional[StateManager] = None
        # Human-in-the-loop gate bookkeeping (hydrated from persisted state)
        self._resume_state: Optional[str] = None
        self._human_decision: Optional[str] = None
        # Invisible cookie-session handshake + chained working memory (hydrated on wake)
        self.session_cookies: Dict[str, str] = {}
        self._working_memory: List[str] = []

    @staticmethod
    def _state_to_dict(state: Any) -> Dict[str, Any]:
        """
        Safely normalizes a persisted state object into a plain dictionary.

        Migration safety net: if a state class comes from an older revision of the
        memory bridge (or a stale module import) that pre-dates a newer structural
        attribute, its own `to_dict()` output simply lacks that key. Reading through
        this dictionary first - and falling back to `vars()`/`__dict__` - lets the
        loader tolerate any attribute mismatch instead of raising AttributeError.
        """
        if isinstance(state, dict):
            return dict(state)

        for method_name in ("to_dict", "model_dump", "as_dict"):
            method = getattr(state, method_name, None)
            if callable(method):
                try:
                    result = method()
                except Exception as exc:  # never let a serialization bug break hydration
                    logger.warning("State %s() failed during hydration: %s", method_name, exc)
                    continue
                if isinstance(result, dict):
                    return dict(result)

        try:
            return dict(vars(state))
        except Exception:
            logger.warning("Could not introspect persisted state of type %s.", type(state).__name__)
            return {}

    def initialize_state(self) -> None:
        """Hydrates previous context from memory bridge or starts fresh WAKE_UP context."""
        logger.info("Agent wake up initialized for Agent ID: '%s'", self.config.agent_id)
        saved_state = self.memory.load_short_term_state(self.config.agent_id)
        self._resume_state = None
        self._human_decision = None

        if saved_state:
            logger.info("Found existing state from previous run (cycle #%s)", getattr(saved_state, "cycle_count", 0))
            # Safe routing for every hydrated field. A state object rebuilt by an older or
            # mismatched memory-bridge revision may predate newer structural attributes
            # (e.g. `session_cookies` / `proposals_ledger`), and a stale module import can
            # leave a partially-populated dataclass. Reading through `getattr` plus the
            # dict representation means migration can never raise AttributeError here.
            saved_dict = self._state_to_dict(saved_state)

            self._resume_state = getattr(saved_state, "current_state", None)
            context_payload = getattr(saved_state, "context_data", None)
            if not isinstance(context_payload, dict):
                fallback_payload = saved_dict.get("context_data")
                context_payload = dict(fallback_payload) if isinstance(fallback_payload, dict) else {}
            self._human_decision = context_payload.get("human_decision")

            raw_cycle = getattr(saved_state, "cycle_count", None)
            if raw_cycle is None:
                raw_cycle = saved_dict.get("cycle_count")
            try:
                cycle_count = int(raw_cycle or 0)
            except (TypeError, ValueError):
                cycle_count = 0

            raw_plan = getattr(saved_state, "active_plan", None)
            if not isinstance(raw_plan, list):
                fallback_plan = saved_dict.get("active_plan")
                raw_plan = fallback_plan if isinstance(fallback_plan, list) else []

            context = AgentContext(
                run_id=f"run_{int(datetime.now(timezone.utc).timestamp())}",
                cycle_count=cycle_count + 1,
                current_state=AgentState.WAKE_UP,
                active_plan=list(raw_plan),
                payload=context_payload,
            )
            # Hydrate the persistent global ledger. Prefer the top-level array; fall back to
            # a ledger mirrored inside context_data for older/newer dashboard shapes.
            ledger = (
                getattr(saved_state, "proposals_ledger", None)
                or saved_dict.get("proposals_ledger")
                or context.payload.get("proposals_ledger")
                or []
            )
            context.proposals_ledger = [
                dict(entry) for entry in ledger if isinstance(entry, dict)
            ] if isinstance(ledger, list) else []
            logger.info("Hydrated proposals_ledger with %d entr(ies) from previous run.", len(context.proposals_ledger))
        else:
            logger.info("No prior state found. Bootstrapping initial agent context.")
            context = AgentContext(
                run_id=f"run_{int(datetime.now(timezone.utc).timestamp())}",
                cycle_count=1,
                current_state=AgentState.WAKE_UP,
            )

        self.state_manager = StateManager(context)

        # Backwards compatibility: promote an old-format single blocking `pending_proposal`
        # (with its top-level `human_decision`) into the global ledger so agent states or
        # dashboards that predate the ledger still hand their task over safely.
        legacy = context.payload.get("pending_proposal")
        if isinstance(legacy, dict):
            migrated = dict(legacy)
            legacy_decision = context.payload.get("human_decision")
            if legacy_decision is not None and not migrated.get("human_decision"):
                migrated["human_decision"] = legacy_decision
            entry = self.state_manager.append_proposal(migrated)
            if migrated.get("human_decision") is not None:
                entry["human_decision"] = migrated.get("human_decision")
            context.payload["pending_proposal"] = entry
            logger.info(
                "Migrated legacy pending_proposal into proposals_ledger as task %s.",
                entry.get("task_id"),
            )

        self.memory.append_long_term_log(
            self.config.agent_id,
            MemoryLogEntry(
                entry_id=f"log_{datetime.now(timezone.utc).timestamp()}",
                event_type="WAKE_UP",
                payload={"cycle_count": self.state_manager.context.cycle_count},
            ),
        )
        # Hydrate the live session cookie map from persisted top-level state.
        # Read through the same safe routing so a state object lacking this attribute
        # (pre-patch revision) degrades to {} rather than raising AttributeError.
        raw_cookies = (
            getattr(saved_state, "session_cookies", None)
            if saved_state else None
        )
        if not isinstance(raw_cookies, dict) and saved_state:
            fallback = self._state_to_dict(saved_state).get("session_cookies")
            raw_cookies = fallback if isinstance(fallback, dict) else {}
        self.session_cookies: Dict[str, str] = {
            str(k): str(v) for k, v in (raw_cookies or {}).items() if str(k).strip()
        }
        # Working memory is per-run (chained tool outputs observed this cycle only).
        self._working_memory = []

    def persist_current_state(self) -> None:
        """Saves current ephemeral context into memory bridge for the next stateless run."""
        if not self.state_manager:
            return

        ctx = self.state_manager.context
        snapshot = ShortTermState(
            run_id=ctx.run_id,
            current_state=ctx.current_state.value,
            cycle_count=ctx.cycle_count,
            active_plan=ctx.active_plan,
            context_data=ctx.payload,
            session_cookies=getattr(self, "session_cookies", {}) or {},
            proposals_ledger=self.state_manager.proposals_ledger,
        )
        # Mirror the ledger into context_data so legacy dashboard builds that read
        # `context_data` still see every task awaiting the human.
        ctx.payload["proposals_ledger"] = self.state_manager.proposals_ledger
        self.memory.save_short_term_state(self.config.agent_id, snapshot)

    def update_session_cookies(self, cookies: Dict[str, str]) -> Dict[str, str]:
        """
        Merges human-provided session cookies into live memory so web reads reuse
        the authenticated session without ever learning passwords. Returns the map.
        """
        clean: Dict[str, str] = {}
        if isinstance(cookies, dict):
            for key, value in cookies.items():
                name = str(key).strip()
                val = str(value).strip()
                if name and val:
                    clean[name] = val
        session = getattr(self, "session_cookies", None)
        if not isinstance(session, dict):
            session = {}
            self.session_cookies = session
        session.update(clean)
        return dict(session)

    def _redacted_session_cookies(self) -> Dict[str, str]:
        """Returns name-only cookie metadata for logs/proposals (values never logged)."""
        session = getattr(self, "session_cookies", {}) or {}
        return {str(k): f"<redacted:{len(str(v))}ch>" for k, v in session.items()} if isinstance(session, dict) else {}

    def generate_monologue(self, prompt: str) -> str:
        """Enforces an internal structured monologue and autonomous reasoning from the AI brain."""
        system_prompt = (
            "You are a fully self-improving autonomous entity. Your objective is to identify digital arbitrage, freelance development tasks, "
            "or automated micro-business opportunities. Browse public boards and tech updates to identify gaps and other ways you can "
            "think of or find online to make money. In the PLAN phase, look at your long-term memory Gist, decide what the most "
            "lucrative or educational digital task is right now, select tools from your toolbox, and execute them in sequence "
            "(up to 5 chained tool calls per cycle). Log your successes and failures to adapt your behavior over time.\n\n"
            "ACCOUNT WALL ROUTING GATE (MANDATORY): You are forbidden from skipping gated work. If you discover a "
            "high-value platform or work opportunity (e.g. Upwork, Freelancer, GitHub Sponsors) that requires "
            "registration, sign-up, or login, you MUST NOT skip it, work around it, or pretend it does not exist. "
            "Instead, immediately route it into an active gate: emit a TOOL_CALL to file_writer_tool "
            "writing 'human_action_needed.json' that contains the platform name, a registration link, and the reason "
            "access is needed. This raises an 'Account Creation & Verification Proposal' so the human approves it "
            "on the dashboard. Treat every login wall, signup form, or email-verification demand as this active "
            "routing gate, never as a reason to abandon the opportunity.\n\n"
            "NON-BLOCKING MULTI-TASK PROPOSAL LEDGER (CRITICAL): Raising a gate NEVER stops your run. The framework "
            "packages the gated action, appends it to a persistent global 'proposals_ledger' with a unique task ID "
            "and a status of PENDING_APPROVAL, and you MUST immediately continue with your next planned pipeline "
            "step in this same cycle. Never wait for the human mid-plan and never end your run early because a "
            "gate was raised: multiple verifications may sit on the dashboard simultaneously while you keep "
            "exploring new opportunities. Only stop after your entire multi-step plan for the cycle is concluded.\n\n"
            "CRITICAL SAFETY BOUNDARY: If an operation requires a legal profile, an invoice, or an account creation step, you must "
            "formulate a structured 'Human Intervention Request' detailing the platform, the reason, and the registration link, write "
            "it to a file named `human_action_needed.json`, and park that specific thread in the proposal ledger until the asset is "
            "provided. NEVER fabricate credentials or attempt to create accounts yourself.\n\n"
            "AVAILABLE TOOLBOX CAPABILITIES:\n"
            "1. web_browser_tool(url: str): Fetches HTML/RSS/text content from public feeds, job boards, or websites. "
            "It automatically injects any human-provided `session_cookies` from memory as a standard Cookie header, "
            "so you reuse your authenticated session automatically.\n"
            "2. file_writer_tool(filepath: str, content: str): Writes scripts, portfolios, reports, or data files in local workspace.\n"
            "3. http_poster_tool(url: str, payload: dict, headers: dict = None): Sends POST requests to simulate platform actions or submit data.\n\n"
            "TOOL CALL FORMAT (CHAIN UP TO 5 SEQUENTIALLY):\n"
            "To invoke tools in your plan, write numbered TOOL_CALL lines in execution order. Each line runs as one "
            "chain link; the output of link N is appended to working memory and visible before link N+1 executes.\n"
            'Example: TOOL_CALL: web_browser_tool {"url": "https://news.example/feed"}\n'
            'Example: TOOL_CALL: file_writer_tool {"filepath": "notes.txt", "content": "Autonomous discovery log"}\n\n'
            "Always structure your output strictly in these sections:\n"
            "THINKING PROCESS:\n"
            "[Your internal analysis of constraints, memory, goals, opportunities, and chosen tool]\n"
            "STEP-BY-STEP PLAN:\n"
            "[Numbered atomic actionable steps, including any TOOL_CALL line]\n"
            "EXECUTION ASSESSMENT:\n"
            "[Evaluation of status, expected outcomes, risks, and next state recommendation]"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.generate_completion(messages)

    def _parse_and_execute_step(self, step_desc: str) -> Dict[str, Any]:
        """
        Parses a step description for TOOL_CALL directives (one or several) and executes
        them as a bounded internal batch, returning a combined result record.
        """
        result: Dict[str, Any] = {
            "step": step_desc,
            "status": "COMPLETED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_calls": [],
        }

        calls = self._extract_tool_calls(step_desc)
        if not calls:
            return result

        budget = max(1, int(getattr(self.config, "max_tool_calls_per_cycle", 5) or 5))
        for tool_name, args in calls[:budget]:
            call_record = self._execute_single_tool_call(tool_name, args)
            result["tool_calls"].append(call_record)
            if call_record.get("status") == "ERROR":
                result["status"] = "ERROR"
        if len(calls) > budget:
            result["status"] = "ERROR"
            result["error"] = (
                f"Tool-chain budget exceeded: planned {len(calls)} TOOL_CALLs, "
                f"executed {budget}. Increase MAX_TOOL_CALLS_PER_CYCLE to run the rest."
            )
            logger.warning("Tool-chain budget exceeded (%d planned, %d executed).", len(calls), budget)
        elif result.get("tool_calls"):
            first = result["tool_calls"][0]
            result["tool_executed"] = first.get("tool_executed")
            result["tool_arguments"] = first.get("tool_arguments")
            result["tool_output"] = first.get("tool_output")

        return result

    @staticmethod
    def _extract_tool_calls(step_desc: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extracts every ordered (tool_name, arguments) TOOL_CALL from a step description,
        supporting full JSON args as well as url-only shorthand for web reads.
        """
        calls: List[Tuple[str, Dict[str, Any]]] = []
        if not step_desc:
            return calls
        for match in re.finditer(r"TOOL_CALL:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})", step_desc):
            try:
                args = json.loads(match.group(2).strip())
                if isinstance(args, dict):
                    calls.append((match.group(1).strip(), args))
            except Exception:
                continue
        if not calls:
            for match in re.finditer(r"([a-zA-Z0-9_]+_tool)\s*(\{.*?\})", step_desc):
                try:
                    args = json.loads(match.group(2).strip())
                    if isinstance(args, dict):
                        calls.append((match.group(1).strip(), args))
                except Exception:
                    continue
        if not calls:
            for match in re.finditer(
                r"([a-zA-Z0-9_]+_tool)\s+(https?://[^\s\"'`]+)", step_desc
            ):
                calls.append((match.group(1).strip(), {"url": match.group(2).strip()}))
        return calls

    def _execute_single_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Runs one chained tool link and returns its normalized result record."""
        record: Dict[str, Any] = {
            "tool_executed": tool_name,
            "tool_arguments": dict(args),
            "status": "COMPLETED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            logger.info("Executing detected tool call: %s with args: %s", tool_name, args)
            if tool_name == "web_browser_tool":
                tool_output = self.toolbox.execute_tool(
                    tool_name,
                    {
                        **args,
                        "session_cookies": args.get("session_cookies")
                        or getattr(self, "session_cookies", {})
                        or None,
                    },
                )
            else:
                tool_output = self.toolbox.execute_tool(tool_name, args)
            record["tool_output"] = tool_output
            record["status"] = tool_output.get("status", "COMPLETED")
            if record["status"] == "ERROR":
                logger.warning("Tool %s executed with error: %s", tool_name, tool_output.get("error"))
        except Exception as parse_err:
            logger.error("Failed to execute tool call '%s': %s", tool_name, parse_err)
            record["status"] = "ERROR"
            record["error"] = f"Failed to execute tool call: {str(parse_err)}"
        return record

    @staticmethod
    def _wrote_human_action_file(result: Dict[str, Any]) -> bool:
        """True when a batched tool result wrote human_action_needed.json (single or multi-call)."""
        if not isinstance(result, dict):
            return False
        for args in [result.get("tool_arguments"), *[c.get("tool_arguments") for c in result.get("tool_calls", []) or []]]:
            if isinstance(args, dict) and str(args.get("filepath", "")).endswith("human_action_needed.json"):
                return True
        return False

    @staticmethod
    def _summarize_tool_output(result: Dict[str, Any]) -> str:
        """
        Compresses one batched step result into a short working-memory note.
        Never includes session cookie values.
        """
        if not isinstance(result, dict):
            return ""
        parts: List[str] = []
        calls = result.get("tool_calls") or []
        if calls:
            for call in calls:
                tool = call.get("tool_executed", "tool")
                output = call.get("tool_output") or {}
                if tool == "web_browser_tool":
                    preview = str(output.get("content_preview", "") or "")[:400]
                    parts.append(f"{tool}: {output.get('status', '?')} {preview}".strip())
                elif tool == "file_writer_tool":
                    parts.append(f"{tool}: wrote {call.get('tool_arguments', {}).get('filepath', '?')}")
                elif tool == "http_poster_tool":
                    parts.append(f"{tool}: {output.get('status', '?')} -> {output.get('url', '?')}")
                else:
                    parts.append(f"{tool}: {call.get('status', '?')}")
        else:
            parts.append(str(result.get("step", ""))[:200])
        summary = " | ".join(p for p in parts if p)[:600]
        return summary

    # ------------------------------------------------------------------
    # Human-in-the-Loop Dashboard Gateway helpers
    # ------------------------------------------------------------------

    # Signals that a task touches external platforms / money / publishing
    APPROVAL_TRIGGER_PATTERNS = (
        r"external platform", r"marketplace", r"payment", r"transaction", r"invoice",
        r"account creation", r"create an account", r"sign ?up", r"registration",
        r"register", r"registration link", r"login wall", r"log ?in", r"sign-?in required",
        r"verify (?:your |email |account|identity)", r"email verification", r"two-?factor",
        r"github sponsors", r"sponsor profile", r"publish", r"submit (?:a )?(?:proposal|listing|pitch)",
        r"\bpitch", r"freelance", r"\bsell\b", r"checkout", r"payout", r"escrow",
        r"\bbid\b", r"human_action_needed", r"human intervention", r"human review",
        r"bank transfer", r"paypal", r"upload (?:asset|listing)", r"post (?:a )?listing",
        r"financial transaction", r"needs (?:an? )?(?:account|verification|approval)",
    )

    _KNOWN_PLATFORMS = (
        "Fiverr", "Upwork", "Toptal", "Gumroad", "Etsy", "GitHub", "GitLab",
        "GitHub Sponsors", "LeetCode", "TaskRabbit", "Udemy", "Product Hunt",
        "Hacker News", "Freelancer", "PeoplePerHour", "Contra", "Legiit",
        "Kwork", "Medium", "Substack", "itch.io", "Stack Overflow", "CodeCanyon",
        "ThemeForest",
    )

    def _requires_human_approval(self, text: str) -> bool:
        """Detects external-platform / financial / publishing intent in a plan step."""
        if not text:
            return False
        low = text.lower()
        if "human_action_needed" in low or "human intervention" in low:
            return True
        return any(re.search(pattern, low) for pattern in self.APPROVAL_TRIGGER_PATTERNS)

    def _classify_proposal_type(self, text: str) -> str:
        low = text.lower()
        if re.search(r"account creation|verification|registration|login|signup|sign-?up|log ?in", low):
            return "Account Creation & Verification"
        if re.search(r"freelance|pitch|bid on|client outreach|proposal to", low):
            return "Freelance Pitch"
        if re.search(r"asset|template|code sale|sell.{0,20}(code|script|plugin|theme)|gumroad|listing", low):
            return "Code Asset Sale"
        return "Trend Report"

    def _extract_target_platform(self, text: str) -> str:
        low = text.lower()
        for platform in self._KNOWN_PLATFORMS:
            if platform.lower() in low:
                return platform
        match = re.search(
            r"(?:platform|marketplace|website|site|on)\s+(?:named\s+)?['\"]?"
            r"([A-Z][A-Za-z0-9]{2,20}(?:\s[A-Z][A-Za-z0-9]{1,20})?)",
            text,
        )
        return match.group(1) if match else "TBD"

    def _extract_estimated_revenue(self, text: str) -> str:
        match = re.search(r"(?:USD|EUR|GBP|\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?", text, re.I)
        if match:
            return match.group(0).strip()
        match = re.search(r"\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP)", text, re.I)
        if match:
            return match.group(0).strip()
        match = re.search(r"\d+\s*(?:per|/)\s*(?:hour|hr|month|mo|week|project|task|day)", text, re.I)
        if match:
            return match.group(0).strip()
        return "TBD"

    def _collect_generated_files(self) -> List[Dict[str, str]]:
        """Collects files written via file_writer_tool during this execution block."""
        files: List[Dict[str, str]] = []
        seen = set()
        if not self.state_manager:
            return files
        for res in self.state_manager.context.execution_results:
            if res.get("tool_executed") != "file_writer_tool" or res.get("status") != "SUCCESS":
                continue
            args = res.get("tool_arguments") or {}
            name = str(args.get("filepath", "unknown_file"))
            if name in seen:
                continue
            seen.add(name)
            files.append({"name": name, "content": str(args.get("content", ""))})
        return files

    def _build_pending_proposal(self, source_text: str, phase: str) -> Dict[str, Any]:
        """
        Builds the structured proposal dictionary that is appended to the
        global `proposals_ledger` for the dashboard. Carries a unique task ID
        and, once stored, a `PENDING_APPROVAL` status.
        """
        stamp = datetime.now(timezone.utc).isoformat()
        run_id = self.state_manager.context.run_id if self.state_manager else "unknown_run"
        task_id = hashlib.sha1(f"{run_id}:{stamp}:{source_text[:80]}".encode("utf-8")).hexdigest()[:12]
        proposal_type = self._classify_proposal_type(source_text)
        proposal: Dict[str, Any] = {
            "task_id": task_id,
            "type": proposal_type,
            "target_platform": self._extract_target_platform(source_text),
            "proposed_action": source_text.strip(),
            "generated_files": self._collect_generated_files(),
            "estimated_revenue": self._extract_estimated_revenue(source_text),
            "detected_in_phase": phase,
            "created_at": stamp,
            "requires_human": True,
            "status": ProposalStatus.PENDING_APPROVAL.value,
            "human_decision": None,
        }
        # Account-wall gate: attach the registration link and session availability.
        if proposal_type == "Account Creation & Verification":
            proposal["requires_account"] = True
            proposal["registration_link"] = self._extract_registration_link(source_text)
            proposal["session_active"] = bool(self.session_cookies)
            proposal["session_cookie_names"] = sorted(self.session_cookies.keys())
        return proposal

    @staticmethod
    def _extract_registration_link(text: str) -> str:
        """Pulls the first http(s) registration/login URL out of a gate request."""
        match = re.search(r"https?://[^\s\"'`\)]+", text)
        return match.group(0).rstrip(".,;") if match else "TBD"

    def _evaluate_human_gate(self) -> str:
        """
        Human-in-the-loop gate evaluated on wake.

        Iterates the `proposals_ledger` and returns a compact verdict:
          NONE     - no ledger entry is currently actionable
          AWAITING - entries exist whose decision is still null
          APPROVED - at least one entry was validated as APPROVED
          DENIED   - at least one entry was validated as DENIED

        Decisions are matched with case-insensitive, whitespace-tolerant string
        checking (`.strip().upper() == "APPROVED"`) so front-end casing shifts are
        handled cleanly. A DENIED verdict takes precedence over APPROVED only in the
        return code; both are processed in the same pass by `_process_proposal_ledger`.
        """
        if not self.state_manager:
            return "NONE"

        ledger = self.state_manager.proposals_ledger
        if not ledger:
            return "NONE"

        approved = False
        denied = False
        awaiting = False

        for entry in ledger:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).strip().upper()
            if status == ProposalStatus.COMPLETED.value:
                continue
            raw_decision = entry.get("human_decision")
            decision = str(raw_decision).strip().upper() if raw_decision is not None else ""
            if decision == "APPROVED":
                approved = True
            elif decision == "DENIED":
                denied = True
            else:
                awaiting = True

        if approved:
            return "APPROVED"
        if denied:
            return "DENIED"
        if awaiting:
            return "AWAITING"
        return "NONE"

    def _detect_ledger_decision(self, entry: Dict[str, Any]) -> str:
        """
        Resolves a single ledger entry's human decision.
        Tolerates front-end casing shifts via `.strip().upper()`.
        """
        raw = entry.get("human_decision")
        if raw is None:
            # Dashboard writes into context_data for the legacy single-task shape.
            ctx = self.state_manager.context if self.state_manager else None
            legacy = ctx.payload.get("human_decision") if ctx else None
            if legacy is not None and len(self.state_manager.proposals_ledger) == 1:
                raw = legacy
        return str(raw).strip().upper() if raw is not None else ""

    def _queue_ledger_task(
        self,
        task: Dict[str, Any],
        origin: str = "HUMAN_APPROVAL",
    ) -> Dict[str, Any]:
        """
        Queues a human-approved ledger task into the active execution window,
        runs its code/payload, and flips its ledger status to COMPLETED.

        The persisted `proposed_action` is replayed through the normal chained
        tool-call executor, so an approved task executes exactly what was reviewed.
        """
        assert self.state_manager is not None
        ctx = self.state_manager.context
        task_id = str(task.get("task_id", "unknown"))
        action_text = str(task.get("proposed_action", "") or task.get("type", "") or "")

        logger.info(
            "Queuing approved task %s into the active execution window (origin=%s).",
            task_id, origin,
        )

        # The approved task runs as its own bounded batch inside the execution window.
        result = self._parse_and_execute_step(action_text or f"1. Execute approved task {task_id}")
        result["task_id"] = task_id
        result["origin"] = origin
        result["approved_task"] = {
            "task_id": task_id,
            "type": task.get("type"),
            "target_platform": task.get("target_platform"),
            "estimated_revenue": task.get("estimated_revenue"),
        }
        ctx.execution_results.append(result)

        summary = self._summarize_tool_output(result)
        if summary:
            self._working_memory.append(f"Approved task {task_id} outcome: {summary}")
            self._working_memory = self._working_memory[-10:]
            ctx.working_memory[:] = self._working_memory

        task["status"] = ProposalStatus.COMPLETED.value
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        task["completion_summary"] = summary[:600]
        task["execution_results"] = result.get("tool_calls", [])

        self.memory.append_long_term_log(
            self.config.agent_id,
            MemoryLogEntry(
                entry_id=f"complete_{task_id}_{datetime.now(timezone.utc).timestamp()}",
                event_type="COMPLETED",
                payload={"task_id": task_id, "origin": origin, "summary": summary[:600]},
            ),
        )
        logger.info("Task %s executed and flipped to COMPLETED.", task_id)
        return result

    def _process_proposal_ledger(self) -> Dict[str, List[str]]:
        """
        Walks the entire `proposals_ledger` in one pass on wake:

          * APPROVED -> print the validation banner, immediately queue that specific
            task into the active execution window, run its payload, flip to COMPLETED.
          * DENIED   -> purge the entry from the active array.
          * null / blank / unknown decision -> left completely untouched, waiting safely
            on the dashboard while the rest of the agent runs independently.

        Returns a tally of what was processed so the caller can decide whether to keep
        running. Never raises into the lifecycle: individual failures are contained.
        """
        tally: Dict[str, List[str]] = {"approved": [], "denied": [], "pending": [], "completed": []}
        if not self.state_manager:
            return tally

        ledger = self.state_manager.proposals_ledger
        if not ledger:
            return tally

        logger.info("Evaluating proposals_ledger (%d entr(ies) present).", len(ledger))

        for entry in list(ledger):
            if not isinstance(entry, dict):
                continue

            task_id = str(entry.get("task_id", "unknown"))
            status = str(entry.get("status", "")).strip().upper()
            if status == ProposalStatus.COMPLETED.value:
                tally["completed"].append(task_id)
                continue

            decision = self._detect_ledger_decision(entry)

            if decision == "APPROVED":
                print(
                    f"[SUCCESS] Human approval validated for task {task_id}. "
                    "Clearing gate and launching task chain..."
                )
                logger.info(
                    "Human approval validated for task %s. Clearing gate and launching task chain...",
                    task_id,
                )
                entry["human_decision"] = "APPROVED"
                try:
                    self._queue_ledger_task(entry, origin="HUMAN_APPROVAL")
                    tally["approved"].append(task_id)
                except Exception as exec_err:
                    logger.error("Approved task %s failed during execution: %s", task_id, exec_err)
                    entry["status"] = ProposalStatus.PENDING_APPROVAL.value
                    entry["last_error"] = str(exec_err)
                    tally["pending"].append(task_id)

            elif decision == "DENIED":
                purged = self.state_manager.remove_proposal(task_id)
                if purged is not None:
                    purged["status"] = ProposalStatus.DENIED.value
                logger.info("Human DENIED task %s; purged from the active ledger array.", task_id)
                self.memory.append_long_term_log(
                    self.config.agent_id,
                    MemoryLogEntry(
                        entry_id=f"denied_{task_id}_{datetime.now(timezone.utc).timestamp()}",
                        event_type="DENIED",
                        payload={"cleared_task_id": task_id, "type": entry.get("type")},
                    ),
                )
                tally["denied"].append(task_id)

            else:
                # null decision (or an unrecognised value): leave it waiting safely.
                tally["pending"].append(task_id)
                logger.debug(
                    "Task %s left untouched in the ledger (decision=%r).", task_id, entry.get("human_decision")
                )

        # Drop execution results that were not attached to a ledger task.
        return tally

    def _find_proposal_for_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Locates an existing ledger entry whose proposed_action matches `text`
        (either direction of containment, whitespace-normalized).
        Used to guarantee one ledger entry per gated action across PLAN and EXECUTE.
        """
        if not self.state_manager or not text:
            return None
        needle = " ".join(str(text).split()).strip()
        if not needle:
            return None
        for entry in self.state_manager.proposals_ledger:
            if not isinstance(entry, dict):
                continue
            haystack = " ".join(str(entry.get("proposed_action", "")).split()).strip()
            if not haystack:
                continue
            if needle in haystack or haystack in needle:
                return entry
        return None

    def _register_gate(self, source_text: str, phase: str) -> Optional[Dict[str, Any]]:
        """
        Non-blocking gate registration with deduplication.

        If the same action is already parked in the ledger (typically detected during
        PLAN and re-observed during EXECUTE), the existing entry is returned and no
        duplicate is appended. Otherwise the new task is appended with a unique ID and
        a PENDING_APPROVAL status and the caller immediately continues its pipeline.
        """
        existing = self._find_proposal_for_action(source_text)
        if existing is not None:
            logger.info(
                "Task %s already parked in proposals_ledger; skipping duplicate gate registration.",
                existing.get("task_id"),
            )
            return existing

        proposal = self._enter_pending_approval(source_text, phase)
        self.notifier.send_alerts(proposal)
        logger.info(
            "Cycle continues: task %s parked in proposals_ledger while remaining steps execute.",
            proposal.get("task_id"),
        )
        return proposal

    def _enter_pending_approval(self, source_text: str, phase: str) -> Dict[str, Any]:
        """
        Packages the human-gated action, appends it to the global `proposals_ledger`
        with a unique ID and a PENDING_APPROVAL status, and returns immediately.

        This is deliberately NON-BLOCKING: the caller keeps executing the rest of its
        planned pipeline steps within the same run instead of halting the lifecycle.
        """
        assert self.state_manager is not None
        ctx = self.state_manager.context
        proposal = self._build_pending_proposal(source_text, phase)

        stored = self.state_manager.append_proposal(proposal)
        # Mirror the newest proposal into context_data so legacy dashboard builds
        # (and the single-task approval flow) continue to render a proposal card.
        ctx.payload["pending_proposal"] = stored
        ctx.payload.setdefault("proposals_ledger", self.state_manager.proposals_ledger)

        self.memory.append_long_term_log(
            self.config.agent_id,
            MemoryLogEntry(
                entry_id=f"pending_{stored.get('task_id')}_{datetime.now(timezone.utc).timestamp()}",
                event_type="PENDING_APPROVAL",
                payload={
                    "pending_proposal": stored,
                    "ledger_depth": len(self.state_manager.proposals_ledger),
                },
            ),
        )
        logger.info(
            "Task %s queued for human approval (type=%s, platform=%s). "
            "Ledger now holds %d entr(ies); continuing pipeline without blocking.",
            stored.get("task_id"), stored.get("type"), stored.get("target_platform"),
            len(self.state_manager.proposals_ledger),
        )
        return stored

    def run_cycle(self) -> None:
        """
        Executes one full lifecycle iteration: WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP.
        Wraps each transition safely so uncaught errors route to the ERROR state
        and save state before shutdown.

        Human-gated work never blocks the run: each gated task is appended to the global
        `proposals_ledger` and the agent immediately proceeds to its next planned pipeline
        step. The FSM settles into SLEEP only once the whole multi-step plan for the cycle
        has concluded.
        """
        try:
            self.initialize_state()
            assert self.state_manager is not None
            ctx = self.state_manager.context

            # --- 0. Human-in-the-Loop ledger reconciliation on wake ---
            # Approved tasks are executed immediately; denied tasks are purged; entries
            # with a null decision stay untouched while the agent runs independently.
            approved_resume = False
            gate = self._evaluate_human_gate()
            if gate in ("APPROVED", "DENIED", "AWAITING"):
                tally = self._process_proposal_ledger()
                approved_resume = bool(tally.get("approved"))
                logger.info(
                    "Ledger reconciliation complete: %d approved, %d denied, %d still awaiting, %d already completed.",
                    len(tally.get("approved", [])),
                    len(tally.get("denied", [])),
                    len(tally.get("pending", [])),
                    len(tally.get("completed", [])),
                )
                if tally.get("pending") and not (approved_resume or tally.get("denied")):
                    self.memory.append_long_term_log(
                        self.config.agent_id,
                        MemoryLogEntry(
                            entry_id=f"await_{datetime.now(timezone.utc).timestamp()}",
                            event_type="PENDING_APPROVAL",
                            payload={
                                "detail": "ledger entries awaiting human decision; continuing pipeline independently",
                                "awaiting_task_ids": tally["pending"],
                            },
                        ),
                    )

            # --- 1. WAKE_UP -> PLAN ---
            resume_reason = "Resuming human-approved task" if approved_resume else "Initial plan formulation"
            self.state_manager.transition_to(AgentState.PLAN, reason=resume_reason)

            monologue = ""
            if approved_resume:
                recent_logs = self.memory.retrieve_recent_logs(self.config.agent_id, limit=5)
                log_summaries = [l.to_dict() for l in recent_logs]
                session_hint = (
                    f"Active session cookies (names only): {sorted(self.session_cookies.keys())}"
                    if self.session_cookies else "No session cookies provided."
                )
                ledger_hint = json.dumps(self.state_manager.ledger_summary())
                plan_prompt = (
                    f"Agent ID: {self.config.agent_id}\n"
                    f"Current Cycle: {ctx.cycle_count}\n"
                    f"Recent Logs: {json.dumps(log_summaries)}\n"
                    f"Proposals Ledger Status: {ledger_hint}\n"
                    f"{session_hint}\n"
                    "Task: A human-approved task chain has just been executed. Continue exploring new "
                    "opportunities and formulate the next atomic operational plan "
                    "(up to 5 chained TOOL_CALL steps)."
                )
                monologue = self.generate_monologue(plan_prompt)
                logger.info("Internal Monologue Generated (post-approval):\n%s", monologue)
                ctx.active_plan = [
                    line.strip() for line in monologue.splitlines()
                    if re.match(r"^\d+\.\s", line.strip())
                ]
                if self._requires_human_approval(monologue):
                    self._register_gate(monologue, "PLAN")
                    # Non-blocking: fall through and keep executing the rest of the plan.
            else:
                recent_logs = self.memory.retrieve_recent_logs(self.config.agent_id, limit=5)
                log_summaries = [l.to_dict() for l in recent_logs]
                session_hint = (
                    f"Active session cookies (names only): {sorted(self.session_cookies.keys())}"
                    if self.session_cookies else "No session cookies provided."
                )
                ledger_hint = json.dumps(self.state_manager.ledger_summary())
                plan_prompt = (
                    f"Agent ID: {self.config.agent_id}\n"
                    f"Current Cycle: {ctx.cycle_count}\n"
                    f"Recent Logs: {json.dumps(log_summaries)}\n"
                    f"Proposals Ledger Status: {ledger_hint}\n"
                    f"{session_hint}\n"
                    "Task: Synthesize context and formulate an atomic operational plan "
                    "(up to 5 chained TOOL_CALL steps)."
                )
                monologue = self.generate_monologue(plan_prompt)
                logger.info("Internal Monologue Generated:\n%s", monologue)
                ctx.active_plan = [
                    line.strip() for line in monologue.splitlines()
                    if re.match(r"^\d+\.\s", line.strip())
                ]

                # PLAN-phase Human-in-the-Loop detection.
                # NON-BLOCKING: the task is appended to the ledger and the run continues.
                if self._requires_human_approval(monologue):
                    self._register_gate(monologue, "PLAN")

            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"plan_{datetime.now(timezone.utc).timestamp()}",
                    event_type="PLAN",
                    payload={"monologue": monologue, "plan": ctx.active_plan},
                ),
            )

            # --- 2. PLAN -> EXECUTE (multi-step batch processing / tool chaining) ---
            self.state_manager.transition_to(AgentState.EXECUTE, reason="Executing plan steps")
            step_limit = min(self.config.max_steps_per_run, MAX_CHAIN_STEPS_PER_CYCLE)
            step_count = min(len(ctx.active_plan), step_limit)
            budget = max(1, min(int(getattr(self.config, "max_tool_calls_per_cycle", MAX_CHAIN_STEPS_PER_CYCLE) or MAX_CHAIN_STEPS_PER_CYCLE), MAX_CHAIN_STEPS_PER_CYCLE))
            logger.info(
                "Executing %d planned step(s) with internal tool-chain budget of %d link(s) "
                "(max %d sequential steps per cycle).",
                step_count, budget, MAX_CHAIN_STEPS_PER_CYCLE,
            )

            triggered_steps: List[str] = []
            tool_calls_used = 0
            chained_working_memory: List[str] = list(getattr(self, "_working_memory", []) or [])
            for idx in range(step_count):
                step_desc = ctx.active_plan[idx]
                logger.info("Running Step [%d/%d]: %s", idx + 1, step_count, step_desc)
                if tool_calls_used >= budget:
                    logger.warning(
                        "Tool-chain budget of %d exhausted at step %d; remaining plan deferred.",
                        budget, idx + 1,
                    )
                    break
                result = self._parse_and_execute_step(step_desc)
                ctx.execution_results.append(result)

                planned_calls = len(self._extract_tool_calls(step_desc))
                observed_calls = len(result.get("tool_calls", [])) or (1 if planned_calls else 0)
                tool_calls_used += max(1, observed_calls)

                # Chain link: observe output into working memory for later links in this run.
                summary = self._summarize_tool_output(result)
                if summary:
                    chained_working_memory.append(f"Step {idx + 1} outcome: {summary}")
                # Keep the last 10 observations only (bounded context growth).
                self._working_memory = chained_working_memory[-10:]
                ctx.working_memory[:] = self._working_memory

                wrote_human_file = self._wrote_human_action_file(result)
                if self._requires_human_approval(step_desc) or wrote_human_file:
                    triggered_steps.append(step_desc)

            logger.info("Tool-chain batch finished: %d link(s) used of %d budgeted.", tool_calls_used, budget)

            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"exec_{datetime.now(timezone.utc).timestamp()}",
                    event_type="ACTION",
                    payload={"results": ctx.execution_results},
                ),
            )

            # EXECUTE-phase Human-in-the-Loop detection.
            # NON-BLOCKING: every newly gated step is appended to the ledger, then the run
            # proceeds. Steps already parked during PLAN are de-duplicated automatically.
            for step_text in triggered_steps:
                self._register_gate(step_text, "EXECUTE")

            # The only condition that parks the FSM in PENDING_APPROVAL is an empty plan:
            # with nothing left to execute, the agent settles on the gate before sleeping.
            if not ctx.active_plan and self.state_manager.pending_proposals():
                self.state_manager.transition_to(
                    AgentState.PENDING_APPROVAL,
                    reason="Plan concluded with %d task(s) awaiting dashboard review" % len(self.state_manager.pending_proposals()),
                )

            # --- 3. EXECUTE -> EVALUATE (chain summary + ledger status included) ---
            self.state_manager.transition_to(AgentState.EVALUATE, reason="Assessing run results")
            chain_notes = ctx.working_memory or getattr(self, "_working_memory", []) or []
            eval_prompt = (
                f"Completed {len(self.state_manager.context.execution_results)} steps.\n"
                f"Execution Results: {json.dumps(self.state_manager.context.execution_results)}\n"
                f"Chained Working Memory: {json.dumps(chain_notes)}\n"
                f"Proposals Ledger Status: {json.dumps(self.state_manager.ledger_summary())}\n"
                "Task: Evaluate if milestones succeeded and confirm safe completion."
            )
            eval_monologue = self.generate_monologue(eval_prompt)
            logger.info("Evaluation Monologue:\n%s", eval_monologue)

            # --- 4. EVALUATE -> SLEEP (whole multi-step plan concluded) ---
            self.state_manager.transition_to(AgentState.SLEEP, reason="Cycle finished gracefully")
            self.persist_current_state()
            logger.info(
                "Agent cycle #%d successfully entered SLEEP mode (ledger: %s).",
                self.state_manager.context.cycle_count,
                self.state_manager.ledger_summary(),
            )

        except Exception as e:
            logger.critical("Fatal error encountered in execution loop: %s", e, exc_info=True)
            if self.state_manager:
                try:
                    self.state_manager.transition_to(AgentState.ERROR, reason=str(e))
                except Exception:
                    self.state_manager.context.current_state = AgentState.ERROR
                self.state_manager.context.errors.append(traceback.format_exc())
                self.memory.append_long_term_log(
                    self.config.agent_id,
                    MemoryLogEntry(
                        entry_id=f"err_{datetime.now(timezone.utc).timestamp()}",
                        event_type="ERROR",
                        payload={"error": str(e), "traceback": traceback.format_exc()},
                    ),
                )
                self.persist_current_state()
                try:
                    self.state_manager.transition_to(AgentState.SLEEP, reason="Post-error graceful wind-down")
                    self.persist_current_state()
                except Exception:
                    pass
            logger.info("Agent error state persisted safely.")


def main() -> None:
    """Entrypoint for standalone execution or GitHub Actions invocation."""
    logger.info("Starting autonomous agent runner...")
    agent = AutonomousAgent()
    agent.run_cycle()
    logger.info("Runner execution concluded.")


if __name__ == "__main__":
    main()

