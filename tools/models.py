"""
Structured Pydantic models for institutional-grade tool outputs.
All tools return structured JSON via these models for reliable LLM reasoning.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ShellResult(BaseModel):
    """Structured output from shell command execution."""
    success: bool = Field(description="Whether the command succeeded (exit code 0)")
    exit_code: int = Field(description="The exit code of the command")
    stdout: str = Field(description="Standard output from the command")
    stderr: str = Field(description="Standard error from the command")


class DelegationResult(BaseModel):
    """Structured output from external coding agent delegation."""
    success: bool = Field(description="Whether the delegated task completed successfully")
    exit_code: int = Field(description="Exit code from the external agent CLI")
    summary: str = Field(description="Human-readable summary of what the agent did")
    files_changed: List[str] = Field(default_factory=list, description="List of files created or modified")
    commands_executed: List[str] = Field(default_factory=list, description="Commands the agent ran internally")
    stdout: str = Field(default="", description="Raw stdout from the agent")
    stderr: str = Field(default="", description="Raw stderr from the agent")
    suggested_next_steps: List[str] = Field(default_factory=list, description="What should happen next")


class VerificationResult(BaseModel):
    """Structured output from verification tools (tests, lint, typecheck)."""
    tool_name: str = Field(description="Name of the verification tool (pytest, flake8, mypy)")
    success: bool = Field(description="Whether verification passed")
    exit_code: int = Field(description="Exit code")
    passed: int = Field(default=0, description="Number of checks/tests passed")
    failed: int = Field(default=0, description="Number of checks/tests failed")
    errors: List[str] = Field(default_factory=list, description="Individual error messages")
    stdout: str = Field(default="", description="Raw stdout")
    stderr: str = Field(default="", description="Raw stderr")


class GitDiffResult(BaseModel):
    """Structured output from git diff inspection."""
    has_changes: bool = Field(description="Whether there are uncommitted changes")
    files_changed: List[str] = Field(default_factory=list, description="List of changed file paths")
    insertions: int = Field(default=0, description="Total lines added")
    deletions: int = Field(default=0, description="Total lines removed")
    diff_content: str = Field(default="", description="The full diff output")
    summary: str = Field(default="", description="Human-readable summary of changes")
