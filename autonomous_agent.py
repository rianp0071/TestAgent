import os
import operator
import subprocess
import ast
import threading
import requests
import json
import logging
from typing import Literal, List, Tuple, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Import modular tools
from tools.models import ShellResult, DelegationResult, VerificationResult, GitDiffResult
from tools.verification import run_tests, run_linter, run_typecheck, get_git_diff
from tools.delegation import antigravity_agent, claude_code_agent, codex_agent

load_dotenv()

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    filename='execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutonomousAgent")

# Also log to console for real-time visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(console_handler)

# ==========================================
# Memory Initialization
# ==========================================
print("Initializing Pinecone Vector DB for Memory...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "test-agent-memory"
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384, # all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# ==========================================
# Core Tools
# ==========================================

hitl_lock = threading.Lock()

def get_hitl_approval(prompt: str) -> bool:
    with hitl_lock:
        approval = input(prompt)
        return approval.lower() in ['y', 'yes']

class SaveMemoryInput(BaseModel):
    information: str = Field(description="The information to remember")

@tool("save_memory", args_schema=SaveMemoryInput)
def save_memory(information: str) -> str:
    """Save an important learning or fact to long-term memory."""
    print(f"\n[Action] Saving to Memory: {information}")
    logger.info(f"Saving to memory: {information}")
    vectorstore.add_texts([information])
    return "Successfully saved to memory."

class SearchMemoryInput(BaseModel):
    query: str = Field(description="The query to search memory for")

@tool("search_memory", args_schema=SearchMemoryInput)
def search_memory(query: str) -> str:
    """Search long-term memory for relevant past learnings."""
    print(f"\n[Action] Searching Memory for: {query}")
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant memories found."
    return "\n".join([d.page_content for d in docs])

class ASTOutlineInput(BaseModel):
    filepath: str = Field(description="The Python file to parse, relative to sandbox_project")

@tool("get_ast_outline", args_schema=ASTOutlineInput)
def get_ast_outline(filepath: str) -> str:
    """Get the structural outline of a Python file (classes and functions) using AST."""
    print(f"\n[Action] Parsing AST for: {filepath}")
    path = os.path.join("sandbox_project", filepath)
    try:
        with open(path, "r") as f:
            tree = ast.parse(f.read())
        
        outline = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                outline.append(f"Function: {node.name} (line {node.lineno})")
            elif isinstance(node, ast.ClassDef):
                outline.append(f"Class: {node.name} (line {node.lineno})")
        return "\n".join(outline) if outline else "No classes or functions found."
    except Exception as e:
        return f"AST Parsing error: {str(e)}"

class ShellCommandInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    purpose: str = Field(description="Why this command is being executed")

@tool("shell_command", args_schema=ShellCommandInput)
def shell_command(command: str, purpose: str) -> str:
    """Execute a shell command and return structured JSON with success, exit_code, stdout, stderr."""
    logger.info(f"Running shell command: {command} | Purpose: {purpose}")
    print(f"\n[Action] Running shell command: `{command}`")
    print(f"[Purpose] {purpose}")
    
    ALLOWED_PREFIXES = ("python", "pytest", "ls", "dir", "echo", "pwd", "cat", "grep", "npm", "node", "git", "pip")
    base_cmd = command.strip().split()[0] if command.strip() else ""
    is_safe = any(base_cmd == p or base_cmd.endswith("/" + p) or base_cmd.endswith("\\" + p) for p in ALLOWED_PREFIXES)
    
    if not is_safe:
        if not get_hitl_approval(f"\n[HITL] Security Warning: Command '{base_cmd}' not in whitelist. Approve? (y/n) > "):
            logger.warning(f"Command rejected by user: {command}")
            return ShellResult(success=False, exit_code=-1, stdout="", stderr="Rejected by sandbox.").model_dump_json()

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd="sandbox_project", timeout=30)
        logger.info(f"Command exit code: {result.returncode}")
        print(f"[Observation] Exit {result.returncode}")
        return ShellResult(
            success=(result.returncode == 0), exit_code=result.returncode,
            stdout=result.stdout.strip(), stderr=result.stderr.strip()
        ).model_dump_json()
    except Exception as e:
        logger.error(f"Shell error: {e}")
        return ShellResult(success=False, exit_code=-1, stdout="", stderr=str(e)).model_dump_json()

class ReadFileInput(BaseModel):
    filepath: str = Field(description="The path to the file to read, relative to sandbox_project")

@tool("read_file", args_schema=ReadFileInput)
def read_file(filepath: str) -> str:
    """Read the contents of a file, returning it with line numbers so you can precisely edit it."""
    print(f"\n[Action] Reading file: {filepath}")
    try:
        path = os.path.join("sandbox_project", filepath)
        with open(path, "r") as f:
            lines = f.readlines()
            numbered_lines = "".join([f"{i+1:03d} | {line}" for i, line in enumerate(lines)])
            print(f"[Observation] Read {len(lines)} lines.")
            return numbered_lines
    except Exception as e:
        err = f"Error reading file: {str(e)}"
        print(f"[Observation] {err}")
        return err

class WriteFileInput(BaseModel):
    filepath: str = Field(description="The path to the file to write, relative to sandbox_project")
    content: str = Field(description="The new full content of the file")

@tool("write_file", args_schema=WriteFileInput)
def write_file(filepath: str, content: str) -> str:
    """Write new content to a file, completely overwriting it."""
    print(f"\n[Action] Writing file: {filepath}")
    if not get_hitl_approval(f"\n[HITL] Approve full overwrite of {filepath}? (y/n) > "):
        print("[Observation] Write rejected.")
        return "Write rejected by human user. Please modify your plan."
    try:
        path = os.path.join("sandbox_project", filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Wrote file: {filepath}")
        print("[Observation] Write successful.")
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        err = f"Error writing file: {str(e)}"
        logger.error(err)
        return err

class ListDirectoryInput(BaseModel):
    directory: str = Field(description="The directory path to list, relative to sandbox_project. Use '.' for root.", default=".")

@tool("list_directory", args_schema=ListDirectoryInput)
def list_directory(directory: str = ".") -> str:
    """List the contents of a directory to understand project structure."""
    print(f"\n[Action] Listing directory: {directory}")
    try:
        path = os.path.join("sandbox_project", directory)
        if not os.path.exists(path):
            return f"Error: Directory '{directory}' does not exist."
        items = os.listdir(path)
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

class SearchCodebaseInput(BaseModel):
    query: str = Field(description="The text to search for")
    directory: str = Field(description="The directory to search in, relative to sandbox_project", default=".")

@tool("search_codebase", args_schema=SearchCodebaseInput)
def search_codebase(query: str, directory: str = ".") -> str:
    """Search for a string in the codebase (.py files only)."""
    print(f"\n[Action] Searching for '{query}' in {directory}")
    path = os.path.join("sandbox_project", directory)
    matches = []
    try:
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if query in line:
                                rel_path = os.path.relpath(filepath, "sandbox_project")
                                matches.append(f"{rel_path}:{i+1}: {line.strip()}")
                except Exception:
                    pass
        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"Error searching codebase: {str(e)}"

class DeleteFileInput(BaseModel):
    filepath: str = Field(description="The path to the file to delete, relative to sandbox_project")

@tool("delete_file", args_schema=DeleteFileInput)
def delete_file(filepath: str) -> str:
    """Delete a file completely."""
    print(f"\n[Action] Deleting file: {filepath}")
    if not get_hitl_approval(f"\n[HITL] Approve DELETION of {filepath}? (y/n) > "):
        return "Deletion rejected by human user."
    try:
        path = os.path.join("sandbox_project", filepath)
        if not os.path.exists(path):
            return "Error: File does not exist."
        os.remove(path)
        logger.info(f"Deleted: {filepath}")
        return f"Successfully deleted {filepath}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"

class ReplaceLinesInput(BaseModel):
    filepath: str = Field(description="The path to the file to edit, relative to sandbox_project")
    start_line: int = Field(description="The starting line number to replace (1-indexed, inclusive)")
    end_line: int = Field(description="The ending line number to replace (1-indexed, inclusive)")
    replacement: str = Field(description="The new content to insert in place of those lines")

@tool("replace_lines", args_schema=ReplaceLinesInput)
def replace_lines(filepath: str, start_line: int, end_line: int, replacement: str) -> str:
    """Safely replace a specific range of lines in a file. Preferred over write_file."""
    print(f"\n[Action] Replacing lines {start_line}-{end_line} in {filepath}")
    if not get_hitl_approval(f"\n[HITL] Approve replacing lines {start_line}-{end_line} in {filepath}? (y/n) > "):
        return "Patch rejected by human user."
    try:
        path = os.path.join("sandbox_project", filepath)
        with open(path, "r") as f:
            lines = f.readlines()
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return f"Error: Invalid line range {start_line}-{end_line} for file with {len(lines)} lines."
        prefix = lines[:start_line - 1]
        suffix = lines[end_line:]
        new_lines = replacement.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        with open(path, "w") as f:
            f.writelines(prefix + new_lines + suffix)
        logger.info(f"Replaced lines {start_line}-{end_line} in {filepath}")
        return f"Successfully replaced lines {start_line}-{end_line} in {filepath}"
    except Exception as e:
        logger.error(f"Replace error: {e}")
        return f"Error editing file: {str(e)}"

class ApplyPatchInput(BaseModel):
    patch_content: str = Field(description="The unified diff patch content to apply")

@tool("apply_patch", args_schema=ApplyPatchInput)
def apply_patch(patch_content: str) -> str:
    """Apply a unified diff patch to modify files."""
    print(f"\n[Action] Applying patch...")
    logger.info("Applying patch")
    if not get_hitl_approval(f"\n[HITL] Approve applying patch? (y/n) > "):
        return "Patch rejected by human user."
    try:
        patch_path = os.path.join("sandbox_project", "temp_patch.diff")
        with open(patch_path, "w") as f:
            f.write(patch_content)
        result = subprocess.run(["git", "apply", "temp_patch.diff"], cwd="sandbox_project", capture_output=True, text=True)
        os.remove(patch_path)
        if result.returncode == 0:
            logger.info("Patch applied successfully.")
            return "Patch applied successfully."
        else:
            return f"Failed to apply patch: {result.stderr}"
    except Exception as e:
        return f"Error applying patch: {str(e)}"

class CreateDirectoryInput(BaseModel):
    directory: str = Field(description="The directory path to create, relative to sandbox_project")

@tool("create_directory", args_schema=CreateDirectoryInput)
def create_directory(directory: str) -> str:
    """Create a new directory or folder structure."""
    print(f"\n[Action] Creating directory: {directory}")
    if not get_hitl_approval(f"\n[HITL] Approve creation of directory {directory}? (y/n) > "):
        return "Directory creation rejected."
    try:
        path = os.path.join("sandbox_project", directory)
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {directory}")
        return f"Successfully created directory {directory}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

class FirecrawlSearchInput(BaseModel):
    query: str = Field(description="The search query to find information on the web")

@tool("firecrawl_search", args_schema=FirecrawlSearchInput)
def firecrawl_search(query: str) -> str:
    """Search the web using Firecrawl for documentation or knowledge."""
    print(f"\n[Action] Web Search (Firecrawl): {query}")
    if not get_hitl_approval(f"\n[HITL] Approve web search for '{query}'? (y/n) > "):
        return "Search rejected by human user."
    try:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return "Error: FIRECRAWL_API_KEY not found."
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"query": query, "pageOptions": {"fetchPageContent": False}}
        response = requests.post("https://api.firecrawl.dev/v1/search", json=payload, headers=headers)
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"
        data = response.json()
        results = data.get("data", [])
        if not results:
            return "No results found."
        formatted = []
        for r in results[:4]:
            formatted.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('description')}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

# ==========================================
# Multi-Agent Architecture (Plan-Execute-Verify)
# ==========================================

class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str

class Plan(BaseModel):
    """A list of steps to accomplish the task."""
    steps: List[str] = Field(description="The steps to follow, in logical order.")

class ReplannerOutput(BaseModel):
    """The output of the replanner."""
    is_complete: bool = Field(description="Whether the entire original task has been successfully completed.")
    final_response: str = Field(description="The final answer or summary to give to the user if complete.", default="")
    new_plan: List[str] = Field(description="The remaining steps to execute if not complete.", default_factory=list)

# Initialize LLM + tool registry
tools = [
    shell_command, read_file, write_file, delete_file, create_directory,
    list_directory, search_codebase, replace_lines, apply_patch, save_memory,
    search_memory, get_ast_outline, firecrawl_search,
    # Delegation tools
    antigravity_agent, claude_code_agent, codex_agent,
    # Verification tools
    run_tests, run_linter, run_typecheck, get_git_diff,
]
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

executor_agent = create_react_agent(llm, tools)

PLANNER_PROMPT = """You are the Planner Agent for an institutional-grade autonomous coding system.
Decompose this task into a step-by-step plan.
Task: {task}

Available capabilities:
- FILESYSTEM: read_file, write_file, replace_lines, apply_patch, delete_file, create_directory, list_directory, search_codebase
- MEMORY: save_memory, search_memory
- ANALYSIS: get_ast_outline, shell_command
- RESEARCH: firecrawl_search
- DELEGATION: antigravity_agent (active), claude_code_agent (inactive), codex_agent (inactive)
- VERIFICATION: run_tests, run_linter, run_typecheck, get_git_diff

Rules:
- Always explore the codebase first and check memory for past learnings.
- For complex tasks, DELEGATE to antigravity_agent then VERIFY with run_tests + get_git_diff.
- After ANY delegation, ALWAYS add verification steps: get_git_diff, run_tests, run_linter.
- If verification fails, add steps to analyze failures and reprompt the agent.
- For modifying existing files, use replace_lines. Always read_file first.
- Before ending, save learnings to memory.
Return ONLY the plan."""

EXECUTOR_PROMPT = (
    "You are the Executor Agent (Orchestrator) of an institutional-grade coding system. "
    "Execute the given step using your tools. You operate inside 'sandbox_project'.\n\n"
    "DELEGATION PROTOCOL:\n"
    "1. When delegating to antigravity_agent, provide clear, specific prompts.\n"
    "2. After delegation, ALWAYS run get_git_diff to inspect what changed.\n"
    "3. ALWAYS run run_tests to verify correctness.\n"
    "4. If tests fail, analyze the failure, then either fix manually or reprompt the agent.\n"
    "5. Log what the external agent changed and why verification passed/failed.\n\n"
    "EDITING PROTOCOL:\n"
    "- Prefer replace_lines or apply_patch over write_file for existing code.\n"
    "- Always read_file first to get exact line numbers.\n\n"
    "SAFETY:\n"
    "- If an action is rejected by user, stop and report failure.\n"
    "- Return a detailed summary including: what changed, verification results, and next steps."
)


def plan_step(state: PlanExecuteState):
    print("\n" + "="*60)
    print("[Planner] Decomposing task into steps...")
    print("="*60)
    task = state["task"]
    logger.info(f"PLAN START: {task}")
    prompt = PLANNER_PROMPT.format(task=task)
    response = llm.with_structured_output(Plan).invoke([HumanMessage(content=prompt)])
    print(f"[Planner] Created Plan:")
    for i, s in enumerate(response.steps):
        print(f"  {i+1}. {s}")
    logger.info(f"PLAN: {response.steps}")
    return {"plan": response.steps, "past_steps": []}


def execute_step(state: PlanExecuteState):
    current_step = state["plan"][0]
    task = state["task"]
    print(f"\n{'='*60}")
    print(f"[Executor] Step: {current_step}")
    print(f"{'='*60}")
    logger.info(f"EXECUTE START: {current_step}")

    sys_prompt = SystemMessage(content=EXECUTOR_PROMPT)
    result = executor_agent.invoke(
        {"messages": [sys_prompt, HumanMessage(content=f"Overall task: {task}\n\nCurrent Step to execute: {current_step}")]}
    )

    summary = result["messages"][-1].content
    logger.info(f"EXECUTE DONE: {current_step} | Summary: {summary[:200]}")
    print(f"\n[Executor] Step completed.\nSummary: {summary}\n")
    return {"past_steps": [(current_step, summary)]}


def replan_step(state: PlanExecuteState):
    print(f"\n{'='*60}")
    print("[Verifier/Replanner] Verifying outcomes...")
    print(f"{'='*60}")
    task = state["task"]
    plan = state["plan"]
    past_steps = state["past_steps"]

    past_steps_str = "\n\n".join([f"Step: {s}\nOutcome: {o}" for s, o in past_steps])

    prompt = f"""You are the Verifier and Replanner Agent for an institutional-grade system.
Original Task: {task}

Past Executed Steps and Outcomes:
{past_steps_str}

Current Remaining Plan (excluding the step just executed):
{plan[1:] if len(plan) > 0 else []}

Verification Rules:
1. If ANY delegation was done, check if verification (tests, diff, lint) was also performed.
2. If verification was NOT performed after delegation, add verification steps.
3. If tests failed, add debugging/fixing steps BEFORE continuing.
4. Only mark complete if ALL tests pass and the task is fully solved.
5. Explain your reasoning for the decision.

Your job:
1. Verify if the original task is completely solved.
2. If solved AND verified, set is_complete=true and write final_response.
3. If NOT solved, return updated new_plan with remaining + new steps.
"""
    logger.info("REPLAN START")
    response = llm.with_structured_output(ReplannerOutput).invoke([HumanMessage(content=prompt)])

    if response.is_complete:
        logger.info(f"TASK COMPLETE: {response.final_response[:200]}")
        print("[Verifier] Task is complete!")
        return {"response": response.final_response, "plan": []}
    else:
        logger.info(f"REPLAN: {response.new_plan}")
        print(f"[Verifier] Task not complete. Updated Plan:")
        for i, s in enumerate(response.new_plan):
            print(f"  {i+1}. {s}")
        return {"plan": response.new_plan}


def should_end(state: PlanExecuteState):
    if "response" in state and state["response"]:
        return "__end__"
    return "execute"

# Assemble the StateGraph
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", plan_step)
workflow.add_node("execute", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "execute")
workflow.add_edge("execute", "replan")
workflow.add_conditional_edges("replan", should_end)

app = workflow.compile()

# ==========================================
# Execution Entrypoint
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Institutional-Grade Autonomous Coding Agent")
    print("  (Plan → Execute → Verify → Replan)")
    print("  Delegation: Antigravity IDE (active)")
    print("  Verification: pytest + flake8 + mypy + git diff")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("\nAgent Task > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            logger.info(f"USER TASK: {user_input}")
            final_state = app.invoke(
                {"task": user_input},
                config={"configurable": {"thread_id": "sandbox_test_level6"}, "recursion_limit": 50}
            )

            print("\n" + "=" * 60)
            print("Final Agent Response:")
            print("=" * 60)
            print(final_state["response"])
            logger.info(f"FINAL RESPONSE: {final_state['response'][:300]}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"RUNTIME ERROR: {e}")
            print(f"\nError: {e}")
