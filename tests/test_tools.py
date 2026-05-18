"""
Institutional Test Suite for the Autonomous Agent.
Tests the tool modules (models, verification, delegation) in isolation.
Run with: python -m pytest tests/ -v
"""
import json
import os
import subprocess
import pytest
import sys

# Add parent dir to path so we can import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.models import ShellResult, DelegationResult, VerificationResult, GitDiffResult


# ==========================================
# Test 1: Structured Model Serialization
# ==========================================

class TestModels:
    def test_shell_result_success(self):
        sr = ShellResult(success=True, exit_code=0, stdout="OK", stderr="")
        data = json.loads(sr.model_dump_json())
        assert data["success"] is True
        assert data["exit_code"] == 0
        assert data["stdout"] == "OK"

    def test_shell_result_failure(self):
        sr = ShellResult(success=False, exit_code=1, stdout="", stderr="error")
        data = json.loads(sr.model_dump_json())
        assert data["success"] is False
        assert data["stderr"] == "error"

    def test_delegation_result_defaults(self):
        dr = DelegationResult(
            success=True, exit_code=0, summary="Done",
        )
        data = json.loads(dr.model_dump_json())
        assert data["files_changed"] == []
        assert data["commands_executed"] == []
        assert data["suggested_next_steps"] == []

    def test_delegation_result_full(self):
        dr = DelegationResult(
            success=True, exit_code=0, summary="Created auth module",
            files_changed=["auth.py", "test_auth.py"],
            commands_executed=["pip install bcrypt"],
            stdout="created files",
            stderr="",
            suggested_next_steps=["run tests"]
        )
        data = json.loads(dr.model_dump_json())
        assert len(data["files_changed"]) == 2
        assert "auth.py" in data["files_changed"]

    def test_verification_result(self):
        vr = VerificationResult(
            tool_name="pytest", success=True, exit_code=0,
            passed=5, failed=0, errors=[], stdout="all passed", stderr=""
        )
        data = json.loads(vr.model_dump_json())
        assert data["tool_name"] == "pytest"
        assert data["passed"] == 5
        assert data["failed"] == 0

    def test_verification_result_with_failures(self):
        vr = VerificationResult(
            tool_name="flake8", success=False, exit_code=1,
            passed=0, failed=3, errors=["E302", "W291", "E501"],
            stdout="", stderr=""
        )
        data = json.loads(vr.model_dump_json())
        assert data["success"] is False
        assert len(data["errors"]) == 3

    def test_git_diff_result_no_changes(self):
        gdr = GitDiffResult(has_changes=False, summary="No changes")
        data = json.loads(gdr.model_dump_json())
        assert data["has_changes"] is False
        assert data["files_changed"] == []

    def test_git_diff_result_with_changes(self):
        gdr = GitDiffResult(
            has_changes=True,
            files_changed=["calculator.py", "test_calculator.py"],
            insertions=10, deletions=3,
            diff_content="+new line\n-old line",
            summary="2 files changed"
        )
        data = json.loads(gdr.model_dump_json())
        assert data["has_changes"] is True
        assert len(data["files_changed"]) == 2
        assert data["insertions"] == 10


# ==========================================
# Test 2: Delegation Output Parser
# ==========================================

class TestDelegationParser:
    def test_parse_success(self):
        from tools.delegation import _parse_delegation_output
        dr = _parse_delegation_output(0, "created auth.py\nmodified test_auth.py", "")
        assert dr.success is True
        assert "auth.py" in dr.files_changed

    def test_parse_failure(self):
        from tools.delegation import _parse_delegation_output
        dr = _parse_delegation_output(1, "", "command not found")
        assert dr.success is False
        assert "failed" in dr.summary.lower()

    def test_parse_empty_output(self):
        from tools.delegation import _parse_delegation_output
        dr = _parse_delegation_output(0, "", "")
        assert dr.success is True
        assert dr.files_changed == []

    def test_parse_suggested_next_steps_on_failure(self):
        from tools.delegation import _parse_delegation_output
        dr = _parse_delegation_output(1, "", "timeout")
        assert len(dr.suggested_next_steps) > 0
        assert any("stderr" in s.lower() or "reprompt" in s.lower() for s in dr.suggested_next_steps)


# ==========================================
# Test 3: Inactive Agent Stubs
# ==========================================

class TestInactiveAgents:
    def test_claude_code_returns_inactive(self):
        from tools.delegation import claude_code_agent
        result = claude_code_agent.invoke({"prompt": "test"})
        data = json.loads(result)
        assert data["success"] is False
        assert "inactive" in data["summary"].lower()

    def test_codex_returns_inactive(self):
        from tools.delegation import codex_agent
        result = codex_agent.invoke({"prompt": "test"})
        data = json.loads(result)
        assert data["success"] is False
        assert "inactive" in data["summary"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
