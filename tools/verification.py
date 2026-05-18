"""
Verification tools: run_tests, run_linter, run_typecheck, get_git_diff.
Each returns structured JSON via Pydantic models.
"""
import os
import re
import subprocess
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tools.models import VerificationResult, GitDiffResult

logger = logging.getLogger("AutonomousAgent")

SANDBOX = "sandbox_project"


class RunTestsInput(BaseModel):
    test_path: str = Field(
        default=".",
        description="Path to test file or directory, relative to sandbox_project. Defaults to running all tests."
    )


@tool("run_tests", args_schema=RunTestsInput)
def run_tests(test_path: str = ".") -> str:
    """Run pytest on the sandbox project and return structured results with pass/fail counts."""
    print(f"\n[Verify] Running pytest on: {test_path}")
    logger.info(f"Running pytest on: {test_path}")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            cwd=SANDBOX,
            capture_output=True,
            text=True,
            timeout=60
        )
        # Parse pass/fail counts from pytest output
        passed = len(re.findall(r" PASSED", result.stdout))
        failed = len(re.findall(r" FAILED", result.stdout))
        errors = re.findall(r"(FAILED .+)", result.stdout)

        vr = VerificationResult(
            tool_name="pytest",
            success=(result.returncode == 0),
            exit_code=result.returncode,
            passed=passed,
            failed=failed,
            errors=errors,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip()
        )
        logger.info(f"pytest result: {passed} passed, {failed} failed, exit={result.returncode}")
        print(f"[Verify] pytest: {passed} passed, {failed} failed")
        return vr.model_dump_json()
    except Exception as e:
        logger.error(f"pytest error: {e}")
        return VerificationResult(
            tool_name="pytest", success=False, exit_code=-1,
            errors=[str(e)], stdout="", stderr=str(e)
        ).model_dump_json()


class RunLinterInput(BaseModel):
    target: str = Field(
        default=".",
        description="File or directory to lint, relative to sandbox_project."
    )


@tool("run_linter", args_schema=RunLinterInput)
def run_linter(target: str = ".") -> str:
    """Run flake8 linter and return structured results with error counts."""
    print(f"\n[Verify] Running flake8 on: {target}")
    logger.info(f"Running flake8 on: {target}")
    try:
        result = subprocess.run(
            ["python", "-m", "flake8", target, "--max-line-length=120"],
            cwd=SANDBOX,
            capture_output=True,
            text=True,
            timeout=30
        )
        error_lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        vr = VerificationResult(
            tool_name="flake8",
            success=(result.returncode == 0),
            exit_code=result.returncode,
            passed=0,
            failed=len(error_lines),
            errors=error_lines[:20],  # Cap at 20 to avoid token explosion
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip()
        )
        logger.info(f"flake8: {len(error_lines)} issues, exit={result.returncode}")
        print(f"[Verify] flake8: {len(error_lines)} issues")
        return vr.model_dump_json()
    except Exception as e:
        logger.error(f"flake8 error: {e}")
        return VerificationResult(
            tool_name="flake8", success=False, exit_code=-1,
            errors=[str(e)], stdout="", stderr=str(e)
        ).model_dump_json()


class RunTypecheckInput(BaseModel):
    target: str = Field(
        default=".",
        description="File or directory to typecheck, relative to sandbox_project."
    )


@tool("run_typecheck", args_schema=RunTypecheckInput)
def run_typecheck(target: str = ".") -> str:
    """Run mypy type checker and return structured results."""
    print(f"\n[Verify] Running mypy on: {target}")
    logger.info(f"Running mypy on: {target}")
    try:
        result = subprocess.run(
            ["python", "-m", "mypy", target, "--ignore-missing-imports"],
            cwd=SANDBOX,
            capture_output=True,
            text=True,
            timeout=60
        )
        error_lines = [l.strip() for l in result.stdout.strip().split("\n") if "error:" in l]
        vr = VerificationResult(
            tool_name="mypy",
            success=(result.returncode == 0),
            exit_code=result.returncode,
            passed=0,
            failed=len(error_lines),
            errors=error_lines[:20],
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip()
        )
        logger.info(f"mypy: {len(error_lines)} errors, exit={result.returncode}")
        print(f"[Verify] mypy: {len(error_lines)} errors")
        return vr.model_dump_json()
    except Exception as e:
        logger.error(f"mypy error: {e}")
        return VerificationResult(
            tool_name="mypy", success=False, exit_code=-1,
            errors=[str(e)], stdout="", stderr=str(e)
        ).model_dump_json()


@tool("get_git_diff")
def get_git_diff() -> str:
    """Inspect uncommitted changes in the sandbox project. Returns structured diff with files changed, insertions, deletions."""
    print("\n[Verify] Running git diff...")
    logger.info("Running git diff")
    try:
        # Get the diff stat
        stat_result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=SANDBOX, capture_output=True, text=True, timeout=10
        )
        # Get changed file names
        names_result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=SANDBOX, capture_output=True, text=True, timeout=10
        )
        # Get full diff (capped)
        diff_result = subprocess.run(
            ["git", "diff"],
            cwd=SANDBOX, capture_output=True, text=True, timeout=10
        )

        files = [f.strip() for f in names_result.stdout.strip().split("\n") if f.strip()]
        diff_text = diff_result.stdout.strip()
        # Cap diff to 3000 chars to avoid token explosion
        if len(diff_text) > 3000:
            diff_text = diff_text[:3000] + "\n... [TRUNCATED]"

        # Parse insertions/deletions from stat
        insertions = len(re.findall(r"^\+", diff_result.stdout, re.MULTILINE))
        deletions = len(re.findall(r"^-", diff_result.stdout, re.MULTILINE))

        has_changes = len(files) > 0
        summary = f"{len(files)} file(s) changed, ~{insertions} insertions, ~{deletions} deletions"

        gdr = GitDiffResult(
            has_changes=has_changes,
            files_changed=files,
            insertions=insertions,
            deletions=deletions,
            diff_content=diff_text,
            summary=summary
        )
        logger.info(f"git diff: {summary}")
        print(f"[Verify] {summary}")
        return gdr.model_dump_json()
    except Exception as e:
        logger.error(f"git diff error: {e}")
        return GitDiffResult(
            has_changes=False, summary=f"Error: {str(e)}"
        ).model_dump_json()
