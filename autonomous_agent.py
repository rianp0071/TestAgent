import os
import operator
import subprocess
import ast
import threading
import requests
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

load_dotenv()

# ==========================================
# Level 3: Memory Initialization
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
# 1. Define Tools
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
    """Execute a shell command and return its stdout, stderr, and exit code."""
    print(f"\n[Action] Running shell command: `{command}`")
    print(f"[Purpose] {purpose}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd="sandbox_project",
            timeout=10
        )
        output = (f"Exit code: {result.returncode}\n"
                  f"Stdout: {result.stdout.strip()}\n"
                  f"Stderr: {result.stderr.strip()}")
        print(f"[Observation] Exit {result.returncode}")
        return output
    except Exception as e:
        err = f"Error executing command: {str(e)}"
        print(f"[Observation] {err}")
        return err

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
    
    # LEVEL 6: Human In The Loop
    if not get_hitl_approval(f"\n[HITL] Approve full overwrite of {filepath}? (y/n) > "):
        print("[Observation] Write rejected.")
        return "Write rejected by human user. Please modify your plan."
        
    try:
        path = os.path.join("sandbox_project", filepath)
        with open(path, "w") as f:
            f.write(content)
        print("[Observation] Write successful.")
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        err = f"Error writing file: {str(e)}"
        print(f"[Observation] {err}")
        return err

class ListDirectoryInput(BaseModel):
    directory: str = Field(description="The directory path to list, relative to sandbox_project. Use '.' for the root of sandbox_project.", default=".")

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
        err = f"Error listing directory: {str(e)}"
        print(f"[Observation] {err}")
        return err

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
        err = f"Error searching codebase: {str(e)}"
        print(f"[Observation] {err}")
        return err

class DeleteFileInput(BaseModel):
    filepath: str = Field(description="The path to the file to delete, relative to sandbox_project")

@tool("delete_file", args_schema=DeleteFileInput)
def delete_file(filepath: str) -> str:
    """Delete a file completely."""
    print(f"\n[Action] Deleting file: {filepath}")
    
    if not get_hitl_approval(f"\n[HITL] Approve DELETION of {filepath}? (y/n) > "):
        print("[Observation] Deletion rejected.")
        return "Deletion rejected by human user. Find another way."
        
    try:
        path = os.path.join("sandbox_project", filepath)
        if not os.path.exists(path):
            return "Error: File does not exist."
        os.remove(path)
        print("[Observation] Deletion successful.")
        return f"Successfully deleted {filepath}"
    except Exception as e:
        err = f"Error deleting file: {str(e)}"
        print(f"[Observation] {err}")
        return err

class ReplaceLinesInput(BaseModel):
    filepath: str = Field(description="The path to the file to edit, relative to sandbox_project")
    start_line: int = Field(description="The starting line number to replace (1-indexed, inclusive)")
    end_line: int = Field(description="The ending line number to replace (1-indexed, inclusive)")
    replacement: str = Field(description="The new content to insert in place of those lines")

@tool("replace_lines", args_schema=ReplaceLinesInput)
def replace_lines(filepath: str, start_line: int, end_line: int, replacement: str) -> str:
    """Safely replace a specific range of lines in a file. This is highly preferred over write_file."""
    print(f"\n[Action] Replacing lines {start_line}-{end_line} in {filepath}")
    
    # LEVEL 6: Human In The Loop
    if not get_hitl_approval(f"\n[HITL] Approve replacing lines {start_line}-{end_line} in {filepath}? (y/n) > "):
        print("[Observation] Patch rejected.")
        return "Patch rejected by human user. Find another way or ask for clarification."
        
    try:
        path = os.path.join("sandbox_project", filepath)
        with open(path, "r") as f:
            lines = f.readlines()
            
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return f"Error: Invalid line range {start_line}-{end_line} for file with {len(lines)} lines."
            
        # Replace the lines (0-indexed)
        prefix = lines[:start_line - 1]
        suffix = lines[end_line:]
        
        # Ensure replacement ends with a newline if the original did
        new_lines = replacement.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
            
        final_lines = prefix + new_lines + suffix
        
        with open(path, "w") as f:
            f.writelines(final_lines)
            
        print("[Observation] Line replacement successful.")
        return f"Successfully replaced lines {start_line}-{end_line} in {filepath}"
    except Exception as e:
        err = f"Error editing file: {str(e)}"
        print(f"[Observation] {err}")
        return err

class CreateDirectoryInput(BaseModel):
    directory: str = Field(description="The directory path to create, relative to sandbox_project")

@tool("create_directory", args_schema=CreateDirectoryInput)
def create_directory(directory: str) -> str:
    """Create a new directory or folder structure."""
    print(f"\n[Action] Creating directory: {directory}")
    if not get_hitl_approval(f"\n[HITL] Approve creation of directory {directory}? (y/n) > "):
        print("[Observation] Directory creation rejected.")
        return "Directory creation rejected by human user."
    try:
        path = os.path.join("sandbox_project", directory)
        os.makedirs(path, exist_ok=True)
        print("[Observation] Directory created successfully.")
        return f"Successfully created directory {directory}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

class FirecrawlSearchInput(BaseModel):
    query: str = Field(description="The search query to find information on the web")

@tool("firecrawl_search", args_schema=FirecrawlSearchInput)
def firecrawl_search(query: str) -> str:
    """Search the web using Firecrawl for documentation or knowledge. Essential for web app development."""
    print(f"\n[Action] Web Search (Firecrawl): {query}")
    if not get_hitl_approval(f"\n[HITL] Approve web search for '{query}'? (y/n) > "):
        print("[Observation] Web search rejected.")
        return "Search rejected by human user."
    try:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return "Error: FIRECRAWL_API_KEY not found in environment."
            
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
        print("[Observation] Web search successful.")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

# ==========================================
# 2. Level 3-6: Multi-Agent Architecture
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
    new_plan: List[str] = Field(description="The remaining steps to execute if not complete. Can include new steps.", default_factory=list)

# Initialize LLM
tools = [
    shell_command, read_file, write_file, delete_file, create_directory, 
    list_directory, search_codebase, replace_lines, save_memory, 
    search_memory, get_ast_outline, firecrawl_search
]
llm = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    temperature=0,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

executor_agent = create_react_agent(llm, tools)

def plan_step(state: PlanExecuteState):
    print("\n[Planner] Decomposing task into steps...")
    task = state["task"]
    prompt = f"""You are the Planner Agent. Decompose this task into a step-by-step plan.
Task: {task}
Remember:
- Make sure to explore the codebase first and check memory for past learnings using search_memory.
- For scaffolding large projects or web apps, use create_directory and write_file.
- For modifying existing files, strictly use replace_lines. Always read_file first to get the exact line numbers.
- To completely remove a file, strictly use delete_file.
- To find documentation or solve complex logic, use firecrawl_search.
- If fixing a bug, explicitly include a step to write tests if none exist, then run tests, then fix the bug, then verify.
- Use get_ast_outline to understand file structure easily.
- Before ending, save your learnings to memory using save_memory.
Return ONLY the plan."""
    response = llm.with_structured_output(Plan).invoke([HumanMessage(content=prompt)])
    print(f"[Planner] Created Plan:")
    for i, s in enumerate(response.steps):
        print(f"  {i+1}. {s}")
    return {"plan": response.steps, "past_steps": []}

def execute_step(state: PlanExecuteState):
    current_step = state["plan"][0]
    task = state["task"]
    print(f"\n[Executor] Executing step: {current_step}")
    
    sys_prompt = SystemMessage(content=(
        "You are the Executor Agent. Your job is to execute the given step of a larger plan using your tools. "
        "You are operating inside the 'sandbox_project' directory.\n"
        "If asked to write tests and none exist, write them dynamically. "
        "Strictly prefer replace_lines for editing existing code. "
        "Strictly use delete_file to remove files. "
        "Always read_file first to get exact line numbers. "
        "If an action is rejected by user, stop and report failure to planner. "
        "When the step is done, return a detailed summary of what you did and the outcomes."
    ))
    
    # Run the executor subgraph
    result = executor_agent.invoke(
        {"messages": [sys_prompt, HumanMessage(content=f"Overall task: {task}\n\nCurrent Step to execute: {current_step}")]}
    )
    
    summary = result["messages"][-1].content
    print(f"\n[Executor] Step completed.\nSummary: {summary}\n")
    
    return {"past_steps": [(current_step, summary)]}

def replan_step(state: PlanExecuteState):
    print("\n[Verifier/Replanner] Verifying outcomes and updating plan...")
    task = state["task"]
    plan = state["plan"]
    past_steps = state["past_steps"]
    
    past_steps_str = "\n\n".join([f"Step: {s}\nOutcome: {o}" for s, o in past_steps])
    
    prompt = f"""You are the Verifier and Replanner Agent.
Original Task: {task}

Past Executed Steps and Outcomes:
{past_steps_str}

Current Remaining Plan (excluding the step just executed):
{plan[1:] if len(plan) > 0 else []}

Your job:
1. Verify if the original task is completely solved based on the outcomes. For example, if tests were supposed to run and pass, did they?
2. If completely solved, set is_complete to true and write a final_response.
3. If NOT solved, or if the current step failed, update the plan. You can add new steps (like 'debug the failure', 'fix the syntax error') or keep the remaining steps. Return the new list of steps in new_plan.
"""
    response = llm.with_structured_output(ReplannerOutput).invoke([HumanMessage(content=prompt)])
    
    if response.is_complete:
        print("[Verifier] Task is complete!")
        return {"response": response.final_response, "plan": []}
    else:
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
# 3. Execution Entrypoint
# ==========================================

if __name__ == "__main__":
    print("==================================================")
    print("Initializing Level 3-6 Codex-Grade Autonomous Agent")
    print("(Memory + Safe Editing + HITL + Planner/Executor)")
    print("==================================================")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\nAgent Task > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            
            # Start the autonomous loop
            final_state = app.invoke(
                {"task": user_input},
                config={"configurable": {"thread_id": "sandbox_test_level6"}, "recursion_limit": 50}
            )
            
            print("\n==================================================")
            print("Final Agent Response:")
            print("==================================================")
            print(final_state["response"])
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
