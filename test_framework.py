"""
Automated unit and integration test suite for the autonomous agent framework.
Validates FSM transitions, state persistence, monologue generation, and error containment.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agent_core import AgentConfig, AutonomousAgent, LLMClient, Toolbox, create_memory_bridge, MAX_CHAIN_STEPS_PER_CYCLE
from cloud_memory_bridge import GitHubGistMemoryBridge
from memory_bridge import LocalFileMemoryBridge, MemoryLogEntry, ShortTermState
from state_manager import AgentContext, AgentState, ProposalStatus, StateManager, StateTransitionError


class TestStateManager(unittest.TestCase):
    def setUp(self):
        self.context = AgentContext(run_id="test_run_1", cycle_count=1)
        self.fsm = StateManager(self.context)

    def test_initial_state(self):
        self.assertEqual(self.fsm.current_state, AgentState.WAKE_UP)

    def test_valid_transitions(self):
        self.fsm.transition_to(AgentState.PLAN, reason="Starting plan")
        self.assertEqual(self.fsm.current_state, AgentState.PLAN)

        self.fsm.transition_to(AgentState.EXECUTE, reason="Executing actions")
        self.assertEqual(self.fsm.current_state, AgentState.EXECUTE)

        self.fsm.transition_to(AgentState.EVALUATE, reason="Assessing outcome")
        self.assertEqual(self.fsm.current_state, AgentState.EVALUATE)

        self.fsm.transition_to(AgentState.SLEEP, reason="Shutting down gracefully")
        self.assertEqual(self.fsm.current_state, AgentState.SLEEP)

    def test_invalid_transition_raises_error(self):
        # Directly transitioning WAKE_UP -> EXECUTE is disallowed by ALLOWED_TRANSITIONS
        # (the pipeline must go through PLAN first).
        with self.assertRaises(StateTransitionError):
            self.fsm.transition_to(AgentState.EXECUTE, reason="Illegal direct leap")

    def test_working_memory_round_trip_and_reset(self):
        self.context.working_memory.append("Step 1 outcome: file_writer_tool wrote a.txt")
        payload = self.context.to_dict()
        self.assertIn("working_memory", payload)
        self.assertEqual(len(payload["working_memory"]), 1)

        restored = AgentContext.from_dict(payload)
        self.assertEqual(restored.working_memory, payload["working_memory"])

        self.fsm.reset_for_new_cycle()
        self.assertEqual(self.context.working_memory, [])
        self.assertEqual(self.context.execution_results, [])
        self.assertEqual(self.context.cycle_count, 2)

    # ------------------------------------------------------------------
    # Non-Blocking Multi-Task Proposal Ledger (FSM layer)
    # ------------------------------------------------------------------

    def test_context_defaults_to_empty_ledger(self):
        self.assertEqual(self.context.proposals_ledger, [])
        self.assertIn("proposals_ledger", self.context.to_dict())

    def test_ledger_round_trip_through_context_serialization(self):
        self.fsm.append_proposal({"task_id": "t1", "type": "Freelance Pitch"})
        payload = self.context.to_dict()
        self.assertEqual(len(payload["proposals_ledger"]), 1)

        restored = AgentContext.from_dict(payload)
        self.assertEqual(restored.proposals_ledger[0]["task_id"], "t1")
        self.assertEqual(restored.proposals_ledger[0]["status"], ProposalStatus.PENDING_APPROVAL.value)

    def test_append_proposal_stamps_unique_id_and_status(self):
        entry = self.fsm.append_proposal({"type": "Trend Report"})
        self.assertEqual(entry["status"], "PENDING_APPROVAL")
        self.assertTrue(entry["task_id"])
        self.assertEqual(len(entry["task_id"]), 12)
        self.assertIsNone(entry["human_decision"])
        self.assertEqual(len(self.fsm.proposals_ledger), 1)

    def test_append_proposal_does_not_duplicate_task_id(self):
        first = self.fsm.append_proposal({"task_id": "dup1", "type": "Freelance Pitch"})
        second = self.fsm.append_proposal({"task_id": "dup1", "type": "Freelance Pitch"})
        self.assertIs(first, second)
        # Several distinct tasks may wait simultaneously.
        self.fsm.append_proposal({"task_id": "dup2"})
        self.assertEqual(len(self.fsm.proposals_ledger), 2)

    def test_remove_and_purge_proposals(self):
        self.fsm.append_proposal({"task_id": "keep"})
        self.fsm.append_proposal({"task_id": "drop"})
        removed = self.fsm.remove_proposal("drop")
        self.assertIsNotNone(removed)
        self.assertEqual([e["task_id"] for e in self.fsm.proposals_ledger], ["keep"])
        self.assertIsNone(self.fsm.remove_proposal("missing"))

        self.fsm.append_proposal({"task_id": "drop2"})
        self.fsm.get_proposal("drop2")["status"] = ProposalStatus.DENIED.value
        purged = self.fsm.purge_proposals("DENIED")
        self.assertEqual(len(purged), 1)
        self.assertEqual([e["task_id"] for e in self.fsm.proposals_ledger], ["keep"])

    def test_pending_proposals_and_ledger_summary(self):
        self.fsm.append_proposal({"task_id": "p1"})
        self.fsm.append_proposal({"task_id": "p2"})
        self.fsm.get_proposal("p2")["status"] = ProposalStatus.COMPLETED.value
        self.assertEqual([e["task_id"] for e in self.fsm.pending_proposals()], ["p1"])
        self.assertEqual(self.fsm.ledger_summary(), {"PENDING_APPROVAL": 1, "COMPLETED": 1})

    def test_ledger_survives_reset_for_new_cycle(self):
        # The ledger is persistent global state: a new cycle must NOT wipe it.
        self.fsm.append_proposal({"task_id": "persist_me"})
        self.fsm.reset_for_new_cycle()
        self.assertEqual(len(self.fsm.proposals_ledger), 1)
        self.assertEqual(self.fsm.proposals_ledger[0]["task_id"], "persist_me")


class TestMemoryBridge(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.bridge = LocalFileMemoryBridge(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_short_term_state(self):
        state = ShortTermState(
            run_id="run_abc",
            current_state=AgentState.PLAN.value,
            cycle_count=4,
            active_plan=["Step 1: Check feed", "Step 2: Act"],
            context_data={"target": "market_watch"},
        )
        self.assertTrue(self.bridge.save_short_term_state("test_agent", state))

        loaded = self.bridge.load_short_term_state("test_agent")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "run_abc")
        self.assertEqual(loaded.cycle_count, 4)
        self.assertEqual(loaded.current_state, AgentState.PLAN.value)
        self.assertEqual(len(loaded.active_plan), 2)
        self.assertEqual(loaded.context_data["target"], "market_watch")

    def test_long_term_log_ledger(self):
        entry = MemoryLogEntry(
            entry_id="log_001",
            event_type="EXECUTION",
            payload={"action": "buy_signal_scan", "status": "ok"},
        )
        self.assertTrue(self.bridge.append_long_term_log("test_agent", entry))
        logs = self.bridge.retrieve_recent_logs("test_agent", limit=10)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].entry_id, "log_001")
        self.assertEqual(logs[0].payload["action"], "buy_signal_scan")


class TestAutonomousAgentCore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.bridge = LocalFileMemoryBridge(storage_dir=self.test_dir)
        self.config = AgentConfig(agent_id="unit_test_agent", max_steps_per_run=3)
        self.agent = AutonomousAgent(config=self.config, memory_bridge=self.bridge)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_cycle_offline_execution(self):
        self.agent.run_cycle()
        self.assertIsNotNone(self.agent.state_manager)
        self.assertEqual(self.agent.state_manager.current_state, AgentState.SLEEP)

        persisted = self.bridge.load_short_term_state("unit_test_agent")
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        self.assertEqual(persisted.cycle_count, 1)

        logs = self.bridge.retrieve_recent_logs("unit_test_agent", limit=10)
        event_types = [l.event_type for l in logs]
        self.assertIn("WAKE_UP", event_types)
        self.assertIn("PLAN", event_types)
        self.assertIn("ACTION", event_types)


class TestLLMFailover(unittest.TestCase):
    def test_ledger_registers_five_providers(self):
        from agent_core import LLMClient
        ledger = LLMClient.FALLBACK_PRIORITY_LEDGER
        self.assertEqual(len(ledger), 5)
        self.assertEqual([p["name"] for p in ledger], ["groq", "gemini", "mistral", "openrouter", "xkiro"])
        self.assertEqual(ledger[0]["model"], "qwen/qwen3.8-27b")
        self.assertEqual(ledger[1]["model"], "gemini-1.5-flash")
        self.assertEqual(ledger[2]["model"], "mistral-small-latest")
        self.assertEqual(ledger[3]["model"], "openrouter/free")
        self.assertEqual(ledger[4]["key_env"], "XKIRO_API_KEY")
        self.assertEqual(ledger[4]["api_base"], "https://api.xkiro.com/v1")

    def test_xkiro_included_when_key_present(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"XKIRO_API_KEY": "xk_test"}, clear=False):
            client = LLMClient(AgentConfig())
            chain = client._resolve_provider_chain()
            names = [p["name"] for p in chain]
            self.assertIn("xkiro", names)

    def test_chain_skips_providers_without_keys(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            os.environ["GROQ_API_KEY"] = "gsk_test"
            client = LLMClient(AgentConfig())
            chain = client._resolve_provider_chain()
            self.assertEqual(len(chain), 1)
            self.assertEqual(chain[0]["name"], "groq")

    def test_failover_on_429_diverts_to_next_provider(self):
        import io
        import urllib.error
        from unittest.mock import patch
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_a", "GEMINI_API_KEY": "gm_b"}, clear=False):
            client = LLMClient(AgentConfig())
            calls = []

            def fake_post(api_base, model, key, messages, temp):
                calls.append((api_base, model))
                if len(calls) == 1:
                    raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, io.BytesIO(b"rate limited"))
                return "recovered via gemini"

            with patch.object(client, "_post_chat_completion", side_effect=fake_post):
                result = client.generate_completion([{"role": "user", "content": "hello"}])
                self.assertEqual(result, "recovered via gemini")
                self.assertEqual(len(calls), 2)
                self.assertIn("groq", calls[0][0])
                self.assertIn("googleapis", calls[1][0])

    def test_all_providers_exhausted_uses_heuristic(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_a", "GEMINI_API_KEY": "gm_b"}, clear=False):
            client = LLMClient(AgentConfig())
            with patch.object(client, "_post_chat_completion", side_effect=TimeoutError("timed out")):
                try:
                    client.generate_completion([{"role": "user", "content": "hello"}])
                    self.fail("Expected TimeoutError when a single provider is exhausted")
                except TimeoutError:
                    pass
            # No keys at all -> offline heuristic path
            with patch.dict(os.environ, {}, clear=True):
                offline = LLMClient(AgentConfig())
                result = offline.generate_completion([{"role": "user", "content": "hello"}])
                self.assertIn("THINKING PROCESS", result)


class TestHumanInTheLoop(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.bridge = LocalFileMemoryBridge(storage_dir=self.test_dir)
        self.config = AgentConfig(agent_id="hitl_agent", max_steps_per_run=3)
        self.agent = AutonomousAgent(
            config=self.config,
            memory_bridge=self.bridge,
            toolbox=Toolbox(workspace_root=self.test_dir),
        )
        # Approval alerts must never perform real network calls during tests.
        self.agent.notifier = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fsm_pending_approval_transitions(self):
        # PLAN / EXECUTE may enter PENDING_APPROVAL
        fsm = StateManager(AgentContext(run_id="r1"))
        fsm.transition_to(AgentState.PLAN, reason="planning")
        fsm.transition_to(AgentState.PENDING_APPROVAL, reason="external task detected")
        self.assertEqual(fsm.current_state, AgentState.PENDING_APPROVAL)
        # PENDING_APPROVAL may wind down to SLEEP
        fsm.transition_to(AgentState.SLEEP, reason="await human")
        self.assertEqual(fsm.current_state, AgentState.SLEEP)

    def test_non_blocking_gate_does_not_halt_the_run(self):
        # The whole point of the ledger: a gated task must NOT stop the lifecycle.
        # EVALUATE -> PENDING_APPROVAL and WAKE_UP -> PENDING_APPROVAL are now legal so a
        # cycle can park tasks mid-flight and keep flowing to SLEEP, while a direct
        # WAKE_UP -> EXECUTE leap is still rejected as out-of-order.
        fsm = StateManager(AgentContext(run_id="nb1"))
        fsm.transition_to(AgentState.PLAN, reason="planning")
        fsm.transition_to(AgentState.EXECUTE, reason="executing")
        fsm.transition_to(AgentState.PENDING_APPROVAL, reason="park gated task")
        fsm.transition_to(AgentState.EVALUATE, reason="keep going")
        fsm.transition_to(AgentState.SLEEP, reason="plan concluded")
        self.assertEqual(fsm.current_state, AgentState.SLEEP)

        fsm2 = StateManager(AgentContext(run_id="nb2"))
        fsm2.transition_to(AgentState.PENDING_APPROVAL, reason="gated work detected on wake")
        self.assertEqual(fsm2.current_state, AgentState.PENDING_APPROVAL)

        # Skipping the PLAN/EXECUTE pipeline entirely remains illegal.
        fsm3 = StateManager(AgentContext(run_id="nb3"))
        with self.assertRaises(StateTransitionError):
            fsm3.transition_to(AgentState.EXECUTE, reason="illegal direct leap")

        # SLEEP is terminal for a run: it may only be re-entered via WAKE_UP.
        fsm4 = StateManager(AgentContext(run_id="nb4"))
        fsm4.transition_to(AgentState.SLEEP, reason="wind down")
        with self.assertRaises(StateTransitionError):
            fsm4.transition_to(AgentState.EXECUTE, reason="illegal post-sleep leap")

    def _seed_ledger(self, ledger, active_plan=None, current_state=AgentState.SLEEP.value):
        """Persists a pre-built proposals_ledger as the previous run's state."""
        self.bridge.save_short_term_state(
            "hitl_agent",
            ShortTermState(
                run_id="run_prev",
                current_state=current_state,
                cycle_count=3,
                active_plan=active_plan or [],
                context_data={},
                proposals_ledger=ledger,
            ),
        )

    @staticmethod
    def _ledger_entry(task_id, decision, action="Submit freelance pitch to Upwork", status="PENDING_APPROVAL"):
        return {
            "task_id": task_id,
            "type": "Freelance Pitch",
            "target_platform": "Upwork",
            "proposed_action": action,
            "generated_files": [],
            "estimated_revenue": "$120",
            "status": status,
            "human_decision": decision,
        }

    def test_run_cycle_is_non_blocking_with_null_decision(self):
        # A pending task with a null decision must NOT halt the run: the agent keeps
        # executing its plan and the FSM still settles into SLEEP.
        self._seed_ledger([self._ledger_entry("await01", None)])
        self.agent.run_cycle()

        persisted = self.bridge.load_short_term_state("hitl_agent")
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        # The task is preserved, untouched, waiting safely on the dashboard.
        self.assertEqual(len(persisted.proposals_ledger), 1)
        self.assertEqual(persisted.proposals_ledger[0]["task_id"], "await01")
        self.assertIsNone(persisted.proposals_ledger[0]["human_decision"])
        self.assertEqual(persisted.proposals_ledger[0]["status"], "PENDING_APPROVAL")
        # The agent DID execute the rest of its pipeline despite the open gate.
        event_types = [l.event_type for l in self.bridge.retrieve_recent_logs("hitl_agent", limit=20)]
        self.assertIn("ACTION", event_types)
        self.assertIn("PLAN", event_types)

    def test_approved_ledger_task_runs_and_flips_to_completed(self):
        self._seed_ledger([
            self._ledger_entry(
                "appr01",
                # Case + whitespace insensitivity is required of the gate.
                "  approved  ",
                action='1. TOOL_CALL: file_writer_tool {"filepath": "approved_deliverable.txt", "content": "ran"}',
            )
        ])
        self.agent.run_cycle()

        persisted = self.bridge.load_short_term_state("hitl_agent")
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        entry = persisted.proposals_ledger[0]
        self.assertEqual(entry["task_id"], "appr01")
        self.assertEqual(entry["status"], "COMPLETED")
        self.assertIsNotNone(entry.get("completed_at"))
        # The approved payload actually executed.
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "approved_deliverable.txt")))
        event_types = [l.event_type for l in self.bridge.retrieve_recent_logs("hitl_agent", limit=20)]
        self.assertIn("COMPLETED", event_types)

    def test_approved_task_prints_validation_banner(self):
        import contextlib
        import io as _io
        self._seed_ledger([self._ledger_entry("appr02", "APPROVED")])
        captured = _io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.agent.run_cycle()
        out = captured.getvalue()
        self.assertIn("[SUCCESS] Human approval validated for task appr02.", out)
        self.assertIn("Clearing gate and launching task chain...", out)

    def test_denied_ledger_task_is_purged(self):
        self._seed_ledger([
            self._ledger_entry("deny01", "DENIED"),
            self._ledger_entry("keep01", None),
        ])
        self.agent.run_cycle()

        persisted = self.bridge.load_short_term_state("hitl_agent")
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        remaining = [e["task_id"] for e in persisted.proposals_ledger]
        self.assertNotIn("deny01", remaining)
        self.assertIn("keep01", remaining)
        event_types = [l.event_type for l in self.bridge.retrieve_recent_logs("hitl_agent", limit=20)]
        self.assertIn("DENIED", event_types)

    def test_mixed_ledger_processed_in_one_pass(self):
        # APPROVED runs, DENIED is purged, null stays untouched - all in the same wake-up.
        self._seed_ledger([
            self._ledger_entry(
                "mix_appr", "APPROVED",
                action='1. TOOL_CALL: file_writer_tool {"filepath": "mixed_ok.txt", "content": "ok"}',
            ),
            self._ledger_entry("mix_deny", "DENIED"),
            self._ledger_entry("mix_wait", None),
        ])
        self.agent.run_cycle()

        persisted = self.bridge.load_short_term_state("hitl_agent")
        by_id = {e["task_id"]: e for e in persisted.proposals_ledger}
        self.assertEqual(by_id["mix_appr"]["status"], "COMPLETED")
        self.assertNotIn("mix_deny", by_id)
        self.assertEqual(by_id["mix_wait"]["status"], "PENDING_APPROVAL")
        self.assertEqual(len(persisted.proposals_ledger), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "mixed_ok.txt")))

    def test_evaluate_human_gate_verdicts(self):
        self.agent.initialize_state()
        self.assertEqual(self.agent._evaluate_human_gate(), "NONE")

        self.agent.state_manager.append_proposal(self._ledger_entry("g_wait", None))
        self.assertEqual(self.agent._evaluate_human_gate(), "AWAITING")

        self.agent.state_manager.get_proposal("g_wait")["human_decision"] = "APPROVED"
        self.assertEqual(self.agent._evaluate_human_gate(), "APPROVED")

        self.agent.state_manager.get_proposal("g_wait")["human_decision"] = "dEnIeD"
        self.assertEqual(self.agent._evaluate_human_gate(), "DENIED")

        # An already-completed entry is not actionable.
        self.agent.state_manager.get_proposal("g_wait")["status"] = "COMPLETED"
        self.assertEqual(self.agent._evaluate_human_gate(), "NONE")

    def test_gate_dedupes_repeated_action_detection(self):
        self.agent.initialize_state()
        text = "1. Submit freelance pitch to Upwork requesting $300 payment"
        first = self.agent._register_gate(text, "PLAN")
        second = self.agent._register_gate(text, "EXECUTE")
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(len(self.agent.state_manager.proposals_ledger), 1)
        # Only one alert fires for the single logical gate.
        self.assertEqual(self.agent.notifier.send_alerts.call_count, 1)

    def test_pending_proposal_structure(self):
        self.agent.initialize_state()
        proposal = self.agent._build_pending_proposal(
            "Submit freelance pitch to Upwork requesting $250 payment milestone", "PLAN"
        )
        for key in ("task_id", "type", "target_platform", "proposed_action", "generated_files", "estimated_revenue"):
            self.assertIn(key, proposal)
        self.assertIsInstance(proposal["task_id"], str)
        self.assertEqual(len(proposal["task_id"]), 12)
        self.assertEqual(proposal["type"], "Freelance Pitch")
        self.assertEqual(proposal["target_platform"], "Upwork")
        self.assertIn("$250", proposal["estimated_revenue"])
        self.assertIsInstance(proposal["generated_files"], list)
        # Entries are born PENDING_APPROVAL with no decision yet.
        self.assertEqual(proposal["status"], "PENDING_APPROVAL")
        self.assertIsNone(proposal["human_decision"])

    def test_approval_trigger_detection(self):
        self.assertTrue(self.agent._requires_human_approval("Submit listing to marketplace and collect payment"))
        self.assertTrue(self.agent._requires_human_approval("write human_action_needed.json for registration"))
        self.assertFalse(self.agent._requires_human_approval("Inspect local workspace files and summarize logs"))

    def test_account_wall_gate_not_skipped(self):
        # High-value platform requiring registration MUST route to a gate, not be skipped.
        wall_text = (
            "Discovered Upwork opportunity but access requires account creation and email "
            "verification: https://www.upwork.com/nx/signup"
        )
        self.assertTrue(self.agent._requires_human_approval(wall_text))
        self.agent.initialize_state()
        proposal = self.agent._build_pending_proposal(wall_text, "PLAN")
        self.assertEqual(proposal["type"], "Account Creation & Verification")
        self.assertTrue(proposal["requires_account"])
        self.assertEqual(proposal["registration_link"], "https://www.upwork.com/nx/signup")
        self.assertEqual(proposal["target_platform"], "Upwork")
        self.assertIn("account creation", proposal["proposed_action"].lower())

    def test_account_wall_routes_to_ledger_end_to_end(self):
        bridge = LocalFileMemoryBridge(storage_dir=tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, bridge.storage_dir, True)
        agent = AutonomousAgent(config=AgentConfig(agent_id="wall_agent"), memory_bridge=bridge)
        agent.notifier = MagicMock()
        agent.generate_monologue = lambda prompt: (
            "THINKING PROCESS:\n- Found high-value GitHub Sponsors lead needing login.\n"
            "STEP-BY-STEP PLAN:\n"
            "1. Account wall on GitHub Sponsors requires registration: https://github.com/sponsors/sign_up\n"
            "2. Defer to human for account creation\n"
            "3. Transition to evaluation\n"
            "EXECUTION ASSESSMENT:\nRouting gate raised; continuing."
        )
        agent.run_cycle()

        agent.notifier.send_alerts.assert_called_once()
        persisted = bridge.load_short_term_state("wall_agent")
        # Non-blocking: the gate is parked in the ledger and the run still concludes.
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        self.assertEqual(len(persisted.proposals_ledger), 1)
        stored = persisted.proposals_ledger[0]
        self.assertEqual(stored["type"], "Account Creation & Verification")
        self.assertTrue(stored["requires_account"])
        self.assertIn("github.com", stored["registration_link"])
        self.assertEqual(stored["status"], "PENDING_APPROVAL")
        self.assertIsNone(stored["human_decision"])

    def test_multi_tool_chain_batch_execution(self):
        # A single step may carry several TOOL_CALL links; all run in one cycle.
        step = (
            '1. TOOL_CALL: file_writer_tool {"filepath": "chain_a.txt", "content": "first"} '
            'TOOL_CALL: file_writer_tool {"filepath": "chain_b.txt", "content": "second"} '
            'TOOL_CALL: file_writer_tool {"filepath": "chain_c.txt", "content": "third"}'
        )
        self.agent.initialize_state()
        result = self.agent._parse_and_execute_step(step)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(len(result["tool_calls"]), 3)
        for call in result["tool_calls"]:
            self.assertEqual(call["status"], "SUCCESS")
        for name in ("chain_a.txt", "chain_b.txt", "chain_c.txt"):
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, name)))

    def test_max_five_sequential_steps_per_cycle(self):
        # The batch window is capped at 5 sequential execution steps per cycle.
        self.assertEqual(MAX_CHAIN_STEPS_PER_CYCLE, 5)
        self.assertEqual(self.agent.config.max_tool_calls_per_cycle, 5)

        plan = [
            f'1. TOOL_CALL: file_writer_tool {{"filepath": "step_{i}.txt", "content": "{i}"}}'
            for i in range(8)
        ]
        self.agent.config.max_steps_per_run = 8
        # Return a plan that is deliberately NOT gate-triggering, so the only bound in
        # play is the five-link batch budget.
        self.agent.generate_monologue = lambda p: (
            "THINKING PROCESS:\n- scan local workspace\n"
            "STEP-BY-STEP PLAN:\n" + "\n".join(plan) + "\n"
            "EXECUTION ASSESSMENT:\nok"
        )
        self.agent.run_cycle()

        written = sorted(f for f in os.listdir(self.test_dir) if f.startswith("step_"))
        # Only 5 sequential links run per cycle; the remainder are deferred.
        self.assertEqual(len(written), 5)
        self.assertEqual(written, [f"step_{i}.txt" for i in range(5)])

        event_types = [l.event_type for l in self.bridge.retrieve_recent_logs("hitl_agent", limit=20)]
        self.assertIn("ACTION", event_types)

    def test_tool_chain_budget_enforced(self):
        # Budget of 2: third call must be deferred with an explicit budget error.
        self.agent.config.max_tool_calls_per_cycle = 2
        self.agent.initialize_state()
        step = (
            '1. TOOL_CALL: file_writer_tool {"filepath": "b1.txt", "content": "1"} '
            'TOOL_CALL: file_writer_tool {"filepath": "b2.txt", "content": "2"} '
            'TOOL_CALL: file_writer_tool {"filepath": "b3.txt", "content": "3"}'
        )
        result = self.agent._parse_and_execute_step(step)
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("budget", result["error"].lower())
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "b3.txt")))

    def test_working_memory_records_chain_outcomes(self):
        self.agent.initialize_state()
        step = '1. TOOL_CALL: file_writer_tool {"filepath": "wm.txt", "content": "payload"}'
        result = self.agent._parse_and_execute_step(step)
        summary = self.agent._summarize_tool_output(result)
        self.assertIn("wm.txt", summary)
        self.assertIn("file_writer_tool", summary)

    def test_session_cookie_hydration_and_persistence(self):
        self.bridge.save_short_term_state(
            "hitl_agent",
            ShortTermState(
                run_id="run_sess",
                current_state=AgentState.SLEEP.value,
                cycle_count=5,
                session_cookies={"sessionid": "abc123", "csrftoken": "xyz789"},
            ),
        )
        self.agent.initialize_state()
        # Hydrated from top-level Gist state on wake
        self.assertEqual(self.agent.session_cookies["sessionid"], "abc123")
        # Persisted back on save (top-level, not inside context_data)
        self.agent.persist_current_state()
        reloaded = self.bridge.load_short_term_state("hitl_agent")
        self.assertEqual(reloaded.session_cookies["csrftoken"], "xyz789")
        self.assertNotIn("session_cookies", reloaded.context_data)

    def test_web_browser_injects_session_cookies(self):
        from unittest.mock import patch
        html_sample = "<html><body>Welcome back, member dashboard</body></html>"
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            mock_resp.read.return_value = html_sample.encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            self.agent.initialize_state()
            self.agent.update_session_cookies({"sessionid": "topsecret", "locale": "en"})
            result = self.agent._parse_and_execute_step(
                '1. TOOL_CALL: web_browser_tool {"url": "https://member.example/dashboard"}'
            )
            self.assertEqual(result["tool_calls"][0]["status"], "SUCCESS")
            request = mock_urlopen.call_args[0][0]
            cookie_header = request.get_header("Cookie")
            self.assertIsNotNone(cookie_header)
            self.assertIn("sessionid=topsecret", cookie_header)
            self.assertIn("locale=en", cookie_header)
            # Redaction helper never exposes raw values in logs/proposals
            redacted = self.agent._redacted_session_cookies()
            self.assertNotIn("topsecret", json.dumps(redacted))
            self.assertIn("sessionid", redacted)

    def test_parse_cookie_string(self):
        parsed = Toolbox.parse_cookie_string(
            'sessionid=abc123; csrftoken="xyz 789"; Secure; HttpOnly; SameSite=Lax'
        )
        self.assertEqual(parsed["sessionid"], "abc123")
        self.assertEqual(parsed["csrftoken"], "xyz 789")
        self.assertNotIn("Secure", parsed)
        self.assertEqual(Toolbox.parse_cookie_string(""), {})

    def test_gist_state_cookies_reach_web_browser_after_wake(self):
        # End-to-end: a top-level `session_cookies` dict in the Gist state must
        # survive hydration and appear as a Cookie literal on the next web read.
        from unittest.mock import patch

        self.bridge.save_short_term_state(
            "hitl_agent",
            ShortTermState(
                run_id="run_gist_cookies",
                current_state=AgentState.SLEEP.value,
                cycle_count=7,
                session_cookies={"sessionid": "gist_sess", "csrftoken": "gist_tok"},
            ),
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            mock_resp.read.return_value = b"<html><body>logged in area</body></html>"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            self.agent.initialize_state()
            self.agent.generate_monologue = lambda p: (
                "THINKING PROCESS:\n- read dashboard\nSTEP-BY-STEP PLAN:\n"
                '1. TOOL_CALL: web_browser_tool {"url": "https://member.example/private"}\n'
                "EXECUTION ASSESSMENT:\nok"
            )
            self.agent.run_cycle()

            cookie_header = mock_urlopen.call_args[0][0].get_header("Cookie")
            self.assertIsNotNone(cookie_header)
            self.assertIn("sessionid=gist_sess", cookie_header)
            self.assertIn("csrftoken=gist_tok", cookie_header)
            # Password material is never stored in state or proposals.
            persisted = self.bridge.load_short_term_state("hitl_agent")
            self.assertNotIn("password", json.dumps(persisted.session_cookies).lower())

    def test_ledger_is_the_persisted_gate_not_legacy_pending_proposal(self):
        # The single blocking pending_proposal is replaced by the global ledger array.
        self._seed_ledger([self._ledger_entry("legacy01", None)])
        self.agent.run_cycle()

        persisted = self.bridge.load_short_term_state("hitl_agent")
        self.assertTrue(hasattr(persisted, "proposals_ledger"))
        self.assertEqual(len(persisted.proposals_ledger), 1)
        self.assertEqual(persisted.proposals_ledger[0]["task_id"], "legacy01")
        # Real dict round-trips through plain JSON without custom encoders.
        round_tripped = ShortTermState.from_dict(json.loads(json.dumps(persisted.to_dict())))
        self.assertEqual(round_tripped.proposals_ledger[0]["task_id"], "legacy01")

    def test_legacy_top_level_pending_proposal_is_migrated_into_ledger(self):
        # Backwards compatibility: an old-format state carrying context_data.pending_proposal
        # must fold into the ledger rather than being silently dropped.
        legacy = self._ledger_entry("oldfmt01", None)
        self.bridge.save_short_term_state(
            "hitl_agent",
            ShortTermState(
                run_id="run_old",
                current_state=AgentState.PENDING_APPROVAL.value,
                cycle_count=4,
                context_data={"pending_proposal": legacy, "human_decision": " APPROVED "},
            ),
        )
        self.agent.run_cycle()

        # The legacy proposal has no ledger status yet, so it starts as PENDING_APPROVAL;
        # the legacy top-level `human_decision` approves and executes it in the same pass.
        persisted = self.bridge.load_short_term_state("hitl_agent")
        by_id = {e["task_id"]: e for e in persisted.proposals_ledger}
        self.assertIn("oldfmt01", by_id)
        self.assertEqual(by_id["oldfmt01"]["status"], "COMPLETED")
        self.assertEqual(by_id["oldfmt01"]["human_decision"], "APPROVED")
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        event_types = [l.event_type for l in self.bridge.retrieve_recent_logs("hitl_agent", limit=20)]
        self.assertIn("COMPLETED", event_types)


class TestNotificationManager(unittest.TestCase):
    PROPOSAL = {
        "task_id": "deadbeef0001",
        "type": "Freelance Pitch",
        "target_platform": "Upwork",
        "proposed_action": "Submit pitch collecting payment",
        "generated_files": [],
        "estimated_revenue": "$250",
    }

    def test_discord_alert_posts_embed(self):
        from unittest.mock import MagicMock, patch
        from agent_core import NotificationManager
        notifier = NotificationManager()
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/1/x"}):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.status = 204
                mock_resp.__enter__.return_value = mock_resp
                mock_urlopen.return_value = mock_resp

                result = notifier.send_discord_alert(self.PROPOSAL)

                self.assertTrue(result)
                mock_urlopen.assert_called_once()
                request = mock_urlopen.call_args[0][0]
                body = json.loads(request.data.decode("utf-8"))
                embed = body["embeds"][0]
                field_values = [f["value"] for f in embed["fields"]]
                self.assertIn("deadbeef0001", field_values)
                self.assertIn("Upwork", field_values)
                self.assertIn("$250", field_values)
                self.assertTrue(any(notifier.dashboard_url in v for v in field_values))

    def test_discord_alert_skipped_without_env(self):
        from unittest.mock import patch
        from agent_core import NotificationManager
        notifier = NotificationManager()
        with patch.dict(os.environ, {}, clear=True):
            with patch("urllib.request.urlopen") as mock_urlopen:
                self.assertFalse(notifier.send_discord_alert(self.PROPOSAL))
                mock_urlopen.assert_not_called()

    def test_email_alert_sent_via_smtp(self):
        from unittest.mock import MagicMock, patch
        from agent_core import NotificationManager
        notifier = NotificationManager()
        env = {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "bot@example.com",
            "SMTP_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env):
            with patch("smtplib.SMTP") as mock_smtp:
                instance = MagicMock()
                mock_smtp.return_value.__enter__.return_value = instance

                result = notifier.send_email_alert(self.PROPOSAL)

                self.assertTrue(result)
                instance.starttls.assert_called_once()
                instance.login.assert_called_once_with("bot@example.com", "secret")
                instance.send_message.assert_called_once()
                sent = instance.send_message.call_args[0][0]
                self.assertIn("deadbeef0001", sent["Subject"])
                self.assertIn("deadbeef0001", sent.get_content())

    def test_email_alert_skipped_without_env(self):
        from unittest.mock import patch
        from agent_core import NotificationManager
        notifier = NotificationManager()
        with patch.dict(os.environ, {}, clear=True):
            with patch("smtplib.SMTP") as mock_smtp:
                self.assertFalse(notifier.send_email_alert(self.PROPOSAL))
                mock_smtp.assert_not_called()

    def test_alerts_fire_when_entering_pending_approval(self):
        from unittest.mock import MagicMock, patch
        bridge = LocalFileMemoryBridge(storage_dir=tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, bridge.storage_dir, True)
        config = AgentConfig(agent_id="notify_agent", max_steps_per_run=3)
        agent = AutonomousAgent(config=config, memory_bridge=bridge)
        agent.notifier = MagicMock()

        # Force a PLAN-phase approval trigger (external platform + payment)
        agent.generate_monologue = lambda prompt: (
            "THINKING PROCESS:\n- Marketplace opportunity detected.\n"
            "STEP-BY-STEP PLAN:\n"
            "1. Submit freelance pitch to Upwork requesting $300 payment\n"
            "2. Track listing status\n"
            "3. Transition to evaluation\n"
            "EXECUTION ASSESSMENT:\nReady for review."
        )
        agent.run_cycle()

        agent.notifier.send_alerts.assert_called_once()
        sent_proposal = agent.notifier.send_alerts.call_args[0][0]
        self.assertIn("task_id", sent_proposal)
        self.assertEqual(sent_proposal["detected_in_phase"], "PLAN")

        persisted = bridge.load_short_term_state("notify_agent")
        # Non-blocking: the gated task is parked in the ledger, yet the cycle
        # still runs to completion and settles into SLEEP.
        self.assertEqual(persisted.current_state, AgentState.SLEEP.value)
        self.assertEqual(len(persisted.proposals_ledger), 1)
        self.assertEqual(persisted.proposals_ledger[0]["task_id"], sent_proposal["task_id"])
        self.assertEqual(persisted.proposals_ledger[0]["status"], "PENDING_APPROVAL")


class TestToolbox(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.toolbox = Toolbox(workspace_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_file_writer_tool_success_and_traversal_prevention(self):
        # 1. Normal successful write
        res = self.toolbox.file_writer_tool("sub/output.txt", "Test content payload")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["bytes_written"], len("Test content payload"))

        # Verify content on disk
        target = os.path.join(self.test_dir, "sub", "output.txt")
        self.assertTrue(os.path.exists(target))
        with open(target, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Test content payload")

        # 2. Directory traversal attempt outside workspace root
        traversal_res = self.toolbox.file_writer_tool("../outside.txt", "Malicious content")
        self.assertEqual(traversal_res["status"], "ERROR")
        self.assertIn("Security violation", traversal_res["error"])

    def test_web_browser_tool_mock(self):
        from unittest.mock import MagicMock, patch

        html_sample = "<html><body><h1>Tech Radar</h1><p>Open jobs for AI Engineers.</p></body></html>"
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            mock_resp.read.return_value = html_sample.encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            res = self.toolbox.web_browser_tool("https://news.ycombinator.com")
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["http_status"], 200)
            self.assertIn("Tech Radar", res["content_preview"])
            self.assertIn("Open jobs for AI Engineers", res["content_preview"])
            # No session cookies supplied -> no Cookie header emitted at all.
            request = mock_urlopen.call_args[0][0]
            self.assertIsNone(request.get_header("Cookie"))

    def test_web_browser_tool_injects_cookie_literals(self):
        # Requirement 3: a top-level `session_cookies` dict is formatted and injected
        # into outgoing request headers as a standard Cookie literal.
        from unittest.mock import MagicMock, patch

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            mock_resp.read.return_value = b"<html><body>member area</body></html>"
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            res = self.toolbox.web_browser_tool(
                "https://member.example/dashboard",
                session_cookies={"sessionid": "abc123", "csrftoken": "xyz789"},
            )
            self.assertEqual(res["status"], "SUCCESS")
            cookie_header = mock_urlopen.call_args[0][0].get_header("Cookie")
            self.assertEqual(cookie_header, "sessionid=abc123; csrftoken=xyz789")

    def test_format_cookie_header_sanitizes_and_skips_blanks(self):
        header = Toolbox._format_cookie_header({"sessionid": "ok", "blank": "", "bad;name": "v", "k": "a; b"})
        self.assertIn("sessionid=ok", header)
        self.assertNotIn("blank", header)
        self.assertNotIn("bad;name", header)
        # Cookie metacharacters are stripped from values so the literal stays valid.
        self.assertIn("k=a b", header)
        self.assertEqual(Toolbox._format_cookie_header(None), "")
        self.assertEqual(Toolbox._format_cookie_header("not-a-dict"), "")

    def test_http_poster_tool_mock(self):
        from unittest.mock import MagicMock, patch

        mock_reply = {"id": "123", "received": True}
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 201
            mock_resp.read.return_value = json.dumps(mock_reply).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            res = self.toolbox.http_poster_tool("https://api.platform.com/submit", {"task": "lead_gen"})
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["http_status"], 201)
            self.assertEqual(res["response"], mock_reply)

    def test_execute_tool_dispatch(self):
        res = self.toolbox.execute_tool(
            "file_writer_tool",
            {"filepath": "dispatch_test.txt", "content": "Toolbox dispatch works"},
        )
        self.assertEqual(res["status"], "SUCCESS")
        unknown = self.toolbox.execute_tool("non_existent_tool", {})
        self.assertEqual(unknown["status"], "ERROR")




class TestCloudMemoryBridge(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.fallback = LocalFileMemoryBridge(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unconfigured_bridge_graceful_fallback(self):
        bridge = GitHubGistMemoryBridge(
            gist_id=None,
            github_token=None,
            fallback_bridge=self.fallback,
        )
        self.assertFalse(bridge.is_configured)

        # Saving and loading should gracefully use fallback without crashing
        state = ShortTermState(
            run_id="run_fallback_1",
            current_state=AgentState.PLAN.value,
            cycle_count=10,
        )
        self.assertTrue(bridge.save_short_term_state("fallback_agent", state))
        loaded = bridge.load_short_term_state("fallback_agent")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "run_fallback_1")
        self.assertEqual(loaded.cycle_count, 10)

    def test_mocked_gist_api_success(self):
        from unittest.mock import MagicMock, patch

        mock_gist_response = {
            "id": "mock_gist_123",
            "files": {
                "cloud_agent_state.json": {
                    "filename": "cloud_agent_state.json",
                    "content": '{"run_id": "run_cloud_9", "current_state": "EXECUTE", "cycle_count": 5, "active_plan": ["task 1"], "context_data": {}, "last_updated": "2026-09-28T12:00:00Z"}'
                }
            }
        }

        bridge = GitHubGistMemoryBridge(
            gist_id="mock_gist_123",
            github_token="ghp_fake_token_for_test",
            fallback_bridge=self.fallback,
        )
        self.assertTrue(bridge.is_configured)

        # Mock GET and PATCH requests
        with patch("urllib.request.urlopen") as mock_urlopen:
            # 1. Test load_short_term_state
            mock_resp_get = MagicMock()
            mock_resp_get.status = 200
            mock_resp_get.read.return_value = json.dumps(mock_gist_response).encode("utf-8")
            mock_resp_get.__enter__.return_value = mock_resp_get

            mock_urlopen.return_value = mock_resp_get
            loaded = bridge.load_short_term_state("cloud_agent")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.run_id, "run_cloud_9")
            self.assertEqual(loaded.current_state, "EXECUTE")
            self.assertEqual(loaded.cycle_count, 5)

            # 2. Test save_short_term_state
            mock_resp_patch = MagicMock()
            mock_resp_patch.status = 200
            mock_resp_patch.read.return_value = b"{}"
            mock_resp_patch.__enter__.return_value = mock_resp_patch

            mock_urlopen.return_value = mock_resp_patch
            save_result = bridge.save_short_term_state("cloud_agent", loaded)
            self.assertTrue(save_result)

    def test_create_memory_bridge_factory_switching(self):
        from agent_core import create_memory_bridge

        # Case 1: Local environment (no cloud credentials)
        local_cfg = AgentConfig(agent_id="test_local", github_token=None, gist_id=None)
        bridge1 = create_memory_bridge(local_cfg)
        self.assertIsInstance(bridge1, LocalFileMemoryBridge)

        # Case 2: Cloud environment (both token and gist_id present)
        cloud_cfg = AgentConfig(agent_id="test_cloud", github_token="ghp_test_token", gist_id="gist_abc_123")
        bridge2 = create_memory_bridge(cloud_cfg)
        self.assertIsInstance(bridge2, GitHubGistMemoryBridge)
        self.assertEqual(bridge2.gist_id, "gist_abc_123")
        self.assertEqual(bridge2.github_token, "ghp_test_token")

if __name__ == "__main__":
    unittest.main()
