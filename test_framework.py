"""
Automated unit and integration test suite for the autonomous agent framework.
Validates FSM transitions, state persistence, monologue generation, and error containment.
"""

import json
import os
import shutil
import tempfile
import unittest

from agent_core import AgentConfig, AutonomousAgent, LLMClient, Toolbox, create_memory_bridge
from cloud_memory_bridge import GitHubGistMemoryBridge
from memory_bridge import LocalFileMemoryBridge, MemoryLogEntry, ShortTermState
from state_manager import AgentContext, AgentState, StateManager, StateTransitionError


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
        with self.assertRaises(StateTransitionError):
            self.fsm.transition_to(AgentState.EXECUTE, reason="Illegal direct leap")


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
