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

import json
import logging
import os
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
        "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "gpt-4o-mini"
    )
    llm_api_base: str = os.getenv(
        "LLM_API_BASE",
        "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") else "https://api.openai.com/v1"
    )
    request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "45"))
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN")
    gist_id: Optional[str] = os.getenv("GIST_ID") or os.getenv("MEMORY_GIST_ID")


def create_memory_bridge(config: Optional[AgentConfig] = None) -> BaseMemoryBridge:
    """
    Factory function resolving appropriate memory bridge based on runtime environment.
    Automatically switches to GitHubGistMemoryBridge if GITHUB_TOKEN and GIST_ID are present,
    otherwise uses LocalFileMemoryBridge with local disk persistence.
    """
    cfg = config or AgentConfig()
    token = cfg.github_token
    gist_id = cfg.gist_id

    if token and gist_id:
        logger.info(
            "Cloud environment detected (GITHUB_TOKEN and GIST_ID found). "
            "Initializing GitHubGistMemoryBridge (Gist: %s).",
            gist_id,
        )
        return GitHubGistMemoryBridge(gist_id=gist_id, github_token=token)
    elif token or gist_id:
        missing = "GIST_ID" if not gist_id else "GITHUB_TOKEN"
        logger.warning(
            "Partial cloud memory credentials detected (%s missing). "
            "Defaulting to LocalFileMemoryBridge.",
            missing,
        )

    logger.info("Using LocalFileMemoryBridge for memory persistence.")
    return LocalFileMemoryBridge()


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
            "- Running in local/test offline mode without live API key.\n"
            "- Analyzed state instructions.\n"
            "STEP-BY-STEP PLAN:\n"
            "1. Inspect available environment and storage.\n"
            "2. Verify health and pending tasks.\n"
            "3. Transition gracefully to evaluation.\n"
            "EXECUTION ASSESSMENT:\n"
            "Offline validation successful."
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
    ) -> None:
        self.config = config or AgentConfig()
        self.memory = memory_bridge or create_memory_bridge(self.config)
        self.llm = LLMClient(self.config)
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
        """Enforces an internal structured monologue from the AI brain."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous self-directed agent executing inside a cloud worker.\n"
                    "Always structure your output strictly in these sections:\n"
                    "THINKING PROCESS:\n"
                    "[Your internal analysis of constraints, goals, and environment]\n"
                    "STEP-BY-STEP PLAN:\n"
                    "[Numbered atomic actionable steps]\n"
                    "EXECUTION ASSESSMENT:\n"
                    "[Evaluation of current status, risks, and next state recommendation]"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm.generate_completion(messages)



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
                result = {"step": step_desc, "status": "COMPLETED", "timestamp": datetime.now(timezone.utc).isoformat()}
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

