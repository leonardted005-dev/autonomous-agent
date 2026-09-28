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
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import os
import re
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_bridge import BaseMemoryBridge, LocalFileMemoryBridge, MemoryLogEntry, ShortTermState
from cloud_memory_bridge import GitHubGistMemoryBridge
from state_manager import AgentContext, AgentState, StateManager

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

    def web_browser_tool(self, url: str) -> Dict[str, Any]:
        """
        Fetches text/HTML/RSS content from web endpoints using standard urllib.request.
        Strips scripts, styles, and markup to return clean textual data.
        """
        logger.info("Toolbox: web_browser_tool called on URL: %s", url)
        if not url or not isinstance(url, str):
            return {"status": "ERROR", "error": "Invalid URL provided: URL must be a non-empty string."}

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"status": "ERROR", "error": f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."}

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AutonomousAgent-Toolbox/1.0 (+https://github.com/leonardted005-dev/autonomous-agent)",
                "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,application/json,*/*",
            },
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
            return self.web_browser_tool(url=arguments.get("url", ""))
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
    ) -> None:
        self.config = config or AgentConfig()
        self.memory = memory_bridge or create_memory_bridge(self.config)
        self.llm = LLMClient(self.config)
        self.toolbox = toolbox or Toolbox()
        self.state_manager: Optional[StateManager] = None
        # Human-in-the-loop gate bookkeeping (hydrated from persisted state)
        self._resume_state: Optional[str] = None
        self._human_decision: Optional[str] = None

    def initialize_state(self) -> None:
        """Hydrates previous context from memory bridge or starts fresh WAKE_UP context."""
        logger.info("Agent wake up initialized for Agent ID: '%s'", self.config.agent_id)
        saved_state = self.memory.load_short_term_state(self.config.agent_id)
        self._resume_state = None
        self._human_decision = None

        if saved_state:
            logger.info("Found existing state from previous run (cycle #%d)", saved_state.cycle_count)
            self._resume_state = saved_state.current_state
            self._human_decision = saved_state.context_data.get("human_decision")
            context = AgentContext(
                run_id=f"run_{int(datetime.now(timezone.utc).timestamp())}",
                cycle_count=saved_state.cycle_count + 1,
                current_state=AgentState.WAKE_UP,
                active_plan=saved_state.active_plan,
                payload=saved_state.context_data,
            )
        else:
            logger.info("No prior state found. Bootstrapping initial agent context.")
            context = AgentContext(
                run_id=f"run_{int(datetime.now(timezone.utc).timestamp())}",
                cycle_count=1,
                current_state=AgentState.WAKE_UP,
            )

        self.state_manager = StateManager(context)
        self.memory.append_long_term_log(
            self.config.agent_id,
            MemoryLogEntry(
                entry_id=f"log_{datetime.now(timezone.utc).timestamp()}",
                event_type="WAKE_UP",
                payload={"cycle_count": self.state_manager.context.cycle_count},
            ),
        )

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
        )
        self.memory.save_short_term_state(self.config.agent_id, snapshot)

    def generate_monologue(self, prompt: str) -> str:
        """Enforces an internal structured monologue and autonomous reasoning from the AI brain."""
        system_prompt = (
            "You are a fully self-improving autonomous entity. Your objective is to identify digital arbitrage, freelance development tasks, "
            "or automated micro-business opportunities. Browse public boards and tech updates to identify gaps and other ways you can "
            "think of or find online to make money. In the PLAN phase, look at your long-term memory Gist, decide what the most "
            "lucrative or educational digital task is right now, select a tool from your toolbox, and execute it. Log your successes "
            "and failures to adapt your behavior over time.\n\n"
            "CRITICAL SAFETY BOUNDARY: If an operation requires a legal profile, an invoice, or an account creation step, you must "
            "formulate a structured 'Human Intervention Request' detailing the platform, the reason, and the registration link, write "
            "it to a file named `human_action_needed.json`, and pause that specific thread until the asset is provided.\n\n"
            "AVAILABLE TOOLBOX CAPABILITIES:\n"
            "1. web_browser_tool(url: str): Fetches HTML/RSS/text content from public feeds, job boards, or websites.\n"
            "2. file_writer_tool(filepath: str, content: str): Writes scripts, portfolios, reports, or data files in local workspace.\n"
            "3. http_poster_tool(url: str, payload: dict, headers: dict = None): Sends POST requests to simulate platform actions or submit data.\n\n"
            "TOOL CALL FORMAT:\n"
            "To invoke a tool in your plan, write a line with this format:\n"
            'TOOL_CALL: <tool_name> {"arg_name": "arg_value", ...}\n'
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
        Parses step description for TOOL_CALL directives or embedded JSON tool commands.
        Executes via Toolbox if detected, returning structured results.
        """
        result: Dict[str, Any] = {
            "step": step_desc,
            "status": "COMPLETED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check for explicit TOOL_CALL directive
        tool_call_match = re.search(r"TOOL_CALL:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})", step_desc)
        if not tool_call_match:
            # Check for generic inline tool invocation: <tool_name>(...) or ```json
            tool_call_match = re.search(r"([a-zA-Z0-9_]+_tool)\s*(\{.*?\})", step_desc)

        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            args_str = tool_call_match.group(2).strip()
            try:
                args = json.loads(args_str)
                logger.info("Executing detected tool call: %s with args: %s", tool_name, args)
                tool_output = self.toolbox.execute_tool(tool_name, args)
                result["tool_executed"] = tool_name
                result["tool_arguments"] = args
                result["tool_output"] = tool_output
                result["status"] = tool_output.get("status", "COMPLETED")
                if result["status"] == "ERROR":
                    logger.warning("Tool %s executed with error: %s", tool_name, tool_output.get("error"))
            except Exception as parse_err:
                logger.error("Failed to parse/execute tool call in step '%s': %s", step_desc, parse_err)
                result["status"] = "ERROR"
                result["error"] = f"Failed to execute tool call: {str(parse_err)}"

        return result

    # ------------------------------------------------------------------
    # Human-in-the-Loop Dashboard Gateway helpers
    # ------------------------------------------------------------------

    # Signals that a task touches external platforms / money / publishing
    APPROVAL_TRIGGER_PATTERNS = (
        r"external platform", r"marketplace", r"payment", r"transaction", r"invoice",
        r"account creation", r"create an account", r"sign ?up", r"registration",
        r"register", r"publish", r"submit (?:a )?(?:proposal|listing|pitch)", r"\bpitch",
        r"freelance", r"\bsell\b", r"checkout", r"payout", r"escrow", r"\bbid\b",
        r"human_action_needed", r"human intervention", r"bank transfer", r"paypal",
        r"upload (?:asset|listing)", r"post (?:a )?listing", r"financial transaction",
    )

    _KNOWN_PLATFORMS = (
        "Fiverr", "Upwork", "Toptal", "Gumroad", "Etsy", "GitHub", "GitLab",
        "LeetCode", "TaskRabbit", "Udemy", "Product Hunt", "Hacker News",
        "Freelancer", "PeoplePerHour", "Contra", "Legiit", "Kwork", "Medium",
        "Substack", "itch.io", "Stack Overflow", "CodeCanyon", "ThemeForest",
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
        """Builds the structured pending_proposal JSON dictionary for the dashboard."""
        stamp = datetime.now(timezone.utc).isoformat()
        run_id = self.state_manager.context.run_id if self.state_manager else "unknown_run"
        task_id = hashlib.sha1(f"{run_id}:{stamp}:{source_text[:80]}".encode("utf-8")).hexdigest()[:12]
        return {
            "task_id": task_id,
            "type": self._classify_proposal_type(source_text),
            "target_platform": self._extract_target_platform(source_text),
            "proposed_action": source_text.strip(),
            "generated_files": self._collect_generated_files(),
            "estimated_revenue": self._extract_estimated_revenue(source_text),
            "detected_in_phase": phase,
            "created_at": stamp,
        }

    def _evaluate_human_gate(self) -> str:
        """
        Human-in-the-loop gate evaluated on wake.
        Returns one of: NONE (no pending task), AWAITING (sleep until decision),
        APPROVED (resume execution), DENIED (clear task and re-plan).
        """
        if not self.state_manager:
            return "NONE"
        ctx = self.state_manager.context
        pending = ctx.payload.get("pending_proposal")
        resumed_pending = self._resume_state == AgentState.PENDING_APPROVAL.value

        if resumed_pending and not pending:
            logger.warning("State was PENDING_APPROVAL but no pending_proposal found; resuming normal planning.")
            return "NONE"
        if not pending:
            return "NONE"

        decision = ctx.payload.get("human_decision")
        if decision is None or str(decision).strip() == "":
            return "AWAITING"
        upper = str(decision).strip().upper()
        if upper == "APPROVED":
            return "APPROVED"
        if upper == "DENIED":
            return "DENIED"
        logger.warning("Unknown human_decision '%s'; treating as awaiting approval.", decision)
        return "AWAITING"

    def _enter_pending_approval(self, source_text: str, phase: str) -> Dict[str, Any]:
        """Formats and persists the pending_proposal, halting the cycle for human review."""
        assert self.state_manager is not None
        ctx = self.state_manager.context
        proposal = self._build_pending_proposal(source_text, phase)
        ctx.payload["pending_proposal"] = proposal
        ctx.payload["human_decision"] = None

        self.memory.append_long_term_log(
            self.config.agent_id,
            MemoryLogEntry(
                entry_id=f"pending_{datetime.now(timezone.utc).timestamp()}",
                event_type="PENDING_APPROVAL",
                payload={"pending_proposal": proposal},
            ),
        )
        logger.info(
            "Task %s requires human approval (type=%s, platform=%s). Persisting pending_proposal.",
            proposal["task_id"], proposal["type"], proposal["target_platform"],
        )
        return proposal

    def run_cycle(self) -> None:
        """
        Executes one full lifecycle iteration: WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP.
        Wraps each transition safely so uncaught errors route to the ERROR state
        and save state before shutdown.
        """
        try:
            self.initialize_state()
            assert self.state_manager is not None
            ctx = self.state_manager.context

            # --- 0. Human-in-the-Loop gate on wake ---
            gate = self._evaluate_human_gate()
            approved_resume = False
            if gate == "AWAITING":
                logger.info("Pending proposal awaits human decision. Sleeping without execution.")
                self.memory.append_long_term_log(
                    self.config.agent_id,
                    MemoryLogEntry(
                        entry_id=f"await_{datetime.now(timezone.utc).timestamp()}",
                        event_type="PENDING_APPROVAL",
                        payload={
                            "detail": "human_decision is null; sleeping until dashboard decision",
                            "pending_proposal": ctx.payload.get("pending_proposal"),
                        },
                    ),
                )
                self.state_manager.transition_to(AgentState.SLEEP, reason="Awaiting human decision on pending proposal")
                self.persist_current_state()
                logger.info("Agent paused in SLEEP awaiting human approval.")
                return

            if gate == "DENIED":
                cleared = ctx.payload.pop("pending_proposal", None)
                ctx.payload.pop("human_decision", None)
                task_ref = cleared.get("task_id") if isinstance(cleared, dict) else "unknown"
                logger.info("Human DENIED pending task %s; clearing task and returning to planning.", task_ref)
                self.memory.append_long_term_log(
                    self.config.agent_id,
                    MemoryLogEntry(
                        entry_id=f"denied_{datetime.now(timezone.utc).timestamp()}",
                        event_type="DENIED",
                        payload={"cleared_task_id": task_ref},
                    ),
                )
            elif gate == "APPROVED":
                cleared = ctx.payload.pop("pending_proposal", None)
                ctx.payload.pop("human_decision", None)
                approved_resume = True
                task_ref = cleared.get("task_id") if isinstance(cleared, dict) else "unknown"
                logger.info("Human APPROVED pending task %s; proceeding to execution block.", task_ref)
                self.memory.append_long_term_log(
                    self.config.agent_id,
                    MemoryLogEntry(
                        entry_id=f"approved_{datetime.now(timezone.utc).timestamp()}",
                        event_type="APPROVED",
                        payload={"approved_task_id": task_ref},
                    ),
                )

            # --- 1. WAKE_UP -> PLAN ---
            resume_reason = "Resuming human-approved task" if approved_resume and ctx.active_plan else "Initial plan formulation"
            self.state_manager.transition_to(AgentState.PLAN, reason=resume_reason)

            if approved_resume and ctx.active_plan:
                monologue = "Resuming human-approved plan:\n" + "\n".join(ctx.active_plan)
                logger.info("Skipping re-plan; executing human-approved active plan.")
            else:
                recent_logs = self.memory.retrieve_recent_logs(self.config.agent_id, limit=5)
                log_summaries = [l.to_dict() for l in recent_logs]
                plan_prompt = (
                    f"Agent ID: {self.config.agent_id}\n"
                    f"Current Cycle: {ctx.cycle_count}\n"
                    f"Recent Logs: {json.dumps(log_summaries)}\n"
                    "Task: Synthesize context and formulate an atomic operational plan."
                )
                monologue = self.generate_monologue(plan_prompt)
                logger.info("Internal Monologue Generated:\n%s", monologue)
                ctx.active_plan = [
                    line.strip() for line in monologue.splitlines() if line.strip().startswith(("1.", "2.", "3."))
                ]

                # PLAN-phase Human-in-the-Loop detection
                if self._requires_human_approval(monologue):
                    self._enter_pending_approval(monologue, "PLAN")
                    self.state_manager.transition_to(
                        AgentState.PENDING_APPROVAL,
                        reason="Plan involves external platform, financial transaction, or asset publishing",
                    )
                    self.persist_current_state()
                    logger.info("Cycle halted in PENDING_APPROVAL state awaiting dashboard review.")
                    return

            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"plan_{datetime.now(timezone.utc).timestamp()}",
                    event_type="PLAN",
                    payload={"monologue": monologue, "plan": ctx.active_plan},
                ),
            )

            # --- 2. PLAN -> EXECUTE ---
            self.state_manager.transition_to(AgentState.EXECUTE, reason="Executing plan steps")
            step_count = min(len(ctx.active_plan), self.config.max_steps_per_run)
            logger.info("Executing %d planned steps for this cycle...", step_count)

            triggered_step: Optional[str] = None
            for idx in range(step_count):
                step_desc = ctx.active_plan[idx]
                logger.info("Running Step [%d/%d]: %s", idx + 1, step_count, step_desc)
                result = self._parse_and_execute_step(step_desc)
                ctx.execution_results.append(result)
                if not approved_resume and triggered_step is None:
                    wrote_human_file = str((result.get("tool_arguments") or {}).get("filepath", "")).endswith(
                        "human_action_needed.json"
                    )
                    if self._requires_human_approval(step_desc) or wrote_human_file:
                        triggered_step = step_desc

            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"exec_{datetime.now(timezone.utc).timestamp()}",
                    event_type="ACTION",
                    payload={"results": ctx.execution_results},
                ),
            )

            # EXECUTE-phase Human-in-the-Loop detection
            if triggered_step is not None:
                self._enter_pending_approval(triggered_step, "EXECUTE")
                self.state_manager.transition_to(
                    AgentState.PENDING_APPROVAL,
                    reason="Execution step involves external platform, financial transaction, or asset publishing",
                )
                self.persist_current_state()
                logger.info("Cycle halted in PENDING_APPROVAL state awaiting dashboard review.")
                return

            # --- 3. EXECUTE -> EVALUATE ---
            self.state_manager.transition_to(AgentState.EVALUATE, reason="Assessing run results")
            eval_prompt = (
                f"Completed {len(self.state_manager.context.execution_results)} steps.\n"
                f"Execution Results: {json.dumps(self.state_manager.context.execution_results)}\n"
                "Task: Evaluate if milestones succeeded and confirm safe completion."
            )
            eval_monologue = self.generate_monologue(eval_prompt)
            logger.info("Evaluation Monologue:\n%s", eval_monologue)

            # --- 4. EVALUATE -> SLEEP ---
            self.state_manager.transition_to(AgentState.SLEEP, reason="Cycle finished gracefully")
            self.persist_current_state()
            logger.info("Agent cycle #%d successfully entered SLEEP mode.", self.state_manager.context.cycle_count)

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

