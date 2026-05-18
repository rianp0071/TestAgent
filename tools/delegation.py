"""
External coding agent delegation tools.
Currently focused on Antigravity IDE integration.
Claude Code and Codex are stubbed for future activation.
"""
import os
import re
import json
import subprocess
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tools.models import DelegationResult

logger = logging.getLogger("AutonomousAgent")

SANDBOX = "sandbox_project"


class DelegatorInput(BaseModel):
    prompt: str = Field(description="The task to delegate to the external coding agent.")


def _parse_delegation_output(returncode: int, stdout: str, stderr: str) -> DelegationResult:
    """Parse raw CLI output into a structured DelegationResult."""
    success = (returncode == 0)

    # Try to extract file changes from output
    files_changed = []
    for pattern in [r"(?:created|modified|wrote|updated)\s+['\"]?([^\s'\"]+\.\w+)", r"^[+-]{3}\s+[ab]/(.+)$"]:
        files_changed.extend(re.findall(pattern, stdout, re.MULTILINE | re.IGNORECASE))
    files_changed = list(set(files_changed))

    # Try to extract commands from output
    commands_executed = re.findall(r"(?:running|executing|ran):\s*[`'\"]?(.+?)[`'\"]?\s*$", stdout, re.MULTILINE | re.IGNORECASE)

    # Generate summary
    if success:
        summary = f"Agent completed successfully. {len(files_changed)} file(s) affected."
    else:
        summary = f"Agent failed (exit {returncode}). Stderr: {stderr[:200]}"

    # Suggest next steps
    next_steps = []
    if files_changed:
        next_steps.append("Inspect changed files with read_file")
        next_steps.append("Run get_git_diff to see exact changes")
        next_steps.append("Run run_tests to verify correctness")
    if not success:
        next_steps.append("Analyze stderr for failure reason")
        next_steps.append("Reprompt the agent with more specific instructions")

    return DelegationResult(
        success=success,
        exit_code=returncode,
        summary=summary,
        files_changed=files_changed,
        commands_executed=commands_executed,
        stdout=stdout[:3000] if len(stdout) > 3000 else stdout,
        stderr=stderr[:1000] if len(stderr) > 1000 else stderr,
        suggested_next_steps=next_steps
    )


@tool("antigravity_agent", args_schema=DelegatorInput)
def antigravity_agent(prompt: str) -> str:
    """Delegate a complex coding task to Antigravity IDE. Returns structured JSON with success, files_changed, and suggested_next_steps."""
    print(f"\n[Delegate] Antigravity IDE: {prompt[:100]}...")
    logger.info(f"DELEGATION [antigravity] prompt: {prompt}")
    try:
        exe_name = "antigravity.cmd" if os.name == "nt" else "antigravity"
        result = subprocess.run(
            [exe_name, "chat", "-m", "agent", prompt],
            cwd=SANDBOX,
            capture_output=True,
            text=True,
            timeout=300
        )
        dr = _parse_delegation_output(result.returncode, result.stdout, result.stderr)
        logger.info(f"DELEGATION [antigravity] result: success={dr.success}, files={dr.files_changed}")
        print(f"[Delegate] Antigravity done: success={dr.success}, files={dr.files_changed}")
        return dr.model_dump_json()
    except subprocess.TimeoutExpired:
        logger.error("DELEGATION [antigravity] TIMEOUT after 300s")
        return DelegationResult(
            success=False, exit_code=-1,
            summary="Antigravity agent timed out after 300 seconds",
            suggested_next_steps=["Break the task into smaller pieces", "Retry with simpler prompt"]
        ).model_dump_json()
    except Exception as e:
        logger.error(f"DELEGATION [antigravity] ERROR: {e}")
        return DelegationResult(
            success=False, exit_code=-1,
            summary=f"Error: {str(e)}",
            suggested_next_steps=["Check if Antigravity CLI is installed", "Verify PATH"]
        ).model_dump_json()


@tool("claude_code_agent", args_schema=DelegatorInput)
def claude_code_agent(prompt: str) -> str:
    """Delegate deep architecture tasks to Claude Code. Returns structured JSON. (Currently inactive — use antigravity_agent instead.)"""
    logger.info(f"DELEGATION [claude_code] prompt: {prompt}")
    return DelegationResult(
        success=False, exit_code=-1,
        summary="Claude Code agent is currently inactive. Use antigravity_agent instead.",
        suggested_next_steps=["Use antigravity_agent for this task"]
    ).model_dump_json()


@tool("codex_agent", args_schema=DelegatorInput)
def codex_agent(prompt: str) -> str:
    """Delegate fast edits to Codex. Returns structured JSON. (Currently inactive — use antigravity_agent instead.)"""
    logger.info(f"DELEGATION [codex] prompt: {prompt}")
    return DelegationResult(
        success=False, exit_code=-1,
        summary="Codex agent is currently inactive. Use antigravity_agent instead.",
        suggested_next_steps=["Use antigravity_agent for this task"]
    ).model_dump_json()
