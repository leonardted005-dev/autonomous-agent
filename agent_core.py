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
        "qwen-2.5-coder-32b" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "gpt-4o-mini"
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
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> Optional[str]:
        if self.config.llm_provider == "groq":
            return os.getenv("GROQ_API_KEY")
        elif self.config.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """
        Invokes OpenAI-compatible chat completion endpoint.
        Returns generated text or a deterministic fallback if API keys are missing or offline.
        """
        if not self.api_key:
            logger.warning(
                "No API key detected for provider '%s'. Simulating heuristic response for offline runner mode.",
                self.config.llm_provider,
            )
            return self._heuristic_fallback(messages)

        endpoint = f"{self.config.llm_api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "StatelessAgentCore/1.0",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode("utf-8", errors="replace")
            logger.error("LLM HTTP error %d: %s", http_err.code, error_body)
            raise RuntimeError(f"LLM API returned HTTP {http_err.code}: {error_body}") from http_err
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            raise

    def _heuristic_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Provides predictable structured responses when no API credentials are provided."""
        last_msg = messages[-1]["content"] if messages else ""
        return (
            "THINKING PROCESS:\n"
            "- Operating in autonomous exploration mode without live API key.\n"
            "- Reviewing long-term memory logs and local workspace environment.\n"
            "- Selecting file_writer_tool to register an autonomous discovery log.\n"
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

    def initialize_state(self) -> None:
        """Hydrates previous context from memory bridge or starts fresh WAKE_UP context."""
        logger.info("Agent wake up initialized for Agent ID: '%s'", self.config.agent_id)
        saved_state = self.memory.load_short_term_state(self.config.agent_id)

        if saved_state:
            logger.info("Found existing state from previous run (cycle #%d)", saved_state.cycle_count)
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
            "You are a fully self-improving autonomous entity. In the PLAN phase, look at your "
            "long-term memory Gist, decide what the most lucrative or educational digital task is right now, "
            "select a tool from your toolbox, and execute it. Log your successes and failures to adapt your behavior over time.\n\n"
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



    def run_cycle(self) -> None:
        """
        Executes one full lifecycle iteration: WAKE_UP -> PLAN -> EXECUTE -> EVALUATE -> SLEEP.
        Wraps each transition safely so uncaught errors route to the ERROR state
        and save state before shutdown.
        """
        try:
            self.initialize_state()
            assert self.state_manager is not None

            # --- 1. WAKE_UP -> PLAN ---
            self.state_manager.transition_to(AgentState.PLAN, reason="Initial plan formulation")
            recent_logs = self.memory.retrieve_recent_logs(self.config.agent_id, limit=5)
            log_summaries = [l.to_dict() for l in recent_logs]

            plan_prompt = (
                f"Agent ID: {self.config.agent_id}\n"
                f"Current Cycle: {self.state_manager.context.cycle_count}\n"
                f"Recent Logs: {json.dumps(log_summaries)}\n"
                "Task: Synthesize context and formulate an atomic operational plan."
            )
            monologue = self.generate_monologue(plan_prompt)
            logger.info("Internal Monologue Generated:\n%s", monologue)

            self.state_manager.context.active_plan = [
                line.strip() for line in monologue.splitlines() if line.strip().startswith(("1.", "2.", "3."))
            ]
            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"plan_{datetime.now(timezone.utc).timestamp()}",
                    event_type="PLAN",
                    payload={"monologue": monologue, "plan": self.state_manager.context.active_plan},
                ),
            )

            # --- 2. PLAN -> EXECUTE ---
            self.state_manager.transition_to(AgentState.EXECUTE, reason="Executing plan steps")
            step_count = min(len(self.state_manager.context.active_plan), self.config.max_steps_per_run)
            logger.info("Executing %d planned steps for this cycle...", step_count)

            for idx in range(step_count):
                step_desc = self.state_manager.context.active_plan[idx]
                logger.info("Running Step [%d/%d]: %s", idx + 1, step_count, step_desc)
                result = self._parse_and_execute_step(step_desc)
                self.state_manager.context.execution_results.append(result)

            self.memory.append_long_term_log(
                self.config.agent_id,
                MemoryLogEntry(
                    entry_id=f"exec_{datetime.now(timezone.utc).timestamp()}",
                    event_type="ACTION",
                    payload={"results": self.state_manager.context.execution_results},
                ),
            )

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

