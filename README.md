# 🤖 Autonomous Multi-Agent AI OS

An advanced, enterprise-grade multi-agent software engineering and deep research system. It leverages stateful orchestration, persistent vector memory, strict line-based safe editing, AST parsing, and robust Human-in-the-Loop (HITL) safeguards to deliver autonomous Codex-level development and institutional-grade research.

---

## 🏗️ System Architecture Overview

This repository houses two cutting-edge multi-agent systems designed to operate at maximum cognitive autonomy:

1. **Codex-Grade Software Engineer (`autonomous_agent.py`)**: A multi-agent developer system optimized for exact codebase edits, testing, CRUD file operations, and folder scaffolding.
2. **Modular Deep Research OS (`deep_research_agent.py`)**: A highly orchestrated, multi-persona research network that compiles, verifies, and memorizes domain-specific intelligence.

---

## 💻 1. Codex-Grade Autonomous Software Engineer (`autonomous_agent.py`)

Designed to operate with surgical precision inside codebases, this agent avoids the common pitfalls of LLM-based coding (such as duplication and accidental deletions) by using a line-anchored, AST-aware execution loop.

### 🌟 Key Architectural Features

*   **Planner-Executor Separation**:
    *   **Planner Node**: Decomposes complex, vague developer tasks into granular, ordered execution plans.
    *   **Executor Node**: Operates as a stateful ReAct agent executing specific tasks utilizing specialized codebase tools.
    *   **Verifier/Replanner Node**: Performs semantic verification (e.g. running test suites), self-corrects on failure, and dynamically updates the task list.
*   **Safe Code Editing (Aider/SWE-Agent style)**:
    *   **Line-Numbered Reading**: Reads files with absolute line numbers, allowing the LLM to understand exact indentation and file layout.
    *   **`replace_lines` Tool**: Safely replaces a precise slice of code from line `X` to line `Y`. Prevents the fragile string-matching errors common in typical LLM code patches.
*   **Abstract Syntax Tree (AST) Parsing**: Inspects Python files to instantly map class definitions, function headers, and starting/ending line boundaries before modifying code.
*   **Thread-Safe Human-in-the-Loop (HITL)**: Protects all write, patch, directory creation, and deletion operations. Implements a strict `threading.Lock()` on console standard input to prevent race conditions during concurrent tool executions.
*   **Long-Term Semantic Memory**: Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and a Serverless Pinecone Vector DB to store and search past development learnings, letting the agent learn from previous debugging loops.
*   **Firecrawl Search**: Connects to the Firecrawl API to search for API documentation, libraries, and best-practice code patterns.

### 🛠️ Developer Agent Tools

*   `read_file`: Reads files with explicit line numbers.
*   `replace_lines`: Safely patches targeted ranges of lines.
*   `write_file`: Overwrites or creates new source files (HITL protected).
*   `delete_file`: Completely removes obsolete code or tests (HITL protected).
*   `create_directory`: Scaffolds folder structures for building full web apps from scratch.
*   `get_ast_outline`: Automatically outlines file structures using syntax tree parsing.
*   `search_codebase`: A secure, pure-Python search tool that avoids binary/pycache encoding crashes.
*   `firecrawl_search`: Searches the web for structural coding advice or documentation.
*   `save_memory` & `search_memory`: Semantic cross-session vector store memory.
*   `shell_command`: Runs pytest, linters, or scripts to verify code correctness.

---

## 🔬 2. Autonomous Modular Research Agent (`deep_research_agent.py`)

A stateful, multi-persona cognitive engine built with LangGraph to distribute, verify, and document deep research queries.

### 🌟 Key Research Features

*   **Parallel Multi-Persona Subgraphs**: fans out to 4 specialized agents:
    *   🏛️ **Government & Policy Agent**: Scraping `.gov`, `.edu`, SEC, and Treasury.
    *   🔬 **Academic Agent**: Sourcing papers from SSRN, arXiv, and PubMed.
    *   💼 **Industry Agent**: Consults Bloomberg, Reuters, and Gartner.
    *   👥 **Community Agent**: Aggregates discussions from Reddit and developer forums.
*   **Adaptive Reflection & Conditional Routing**: Analyzes research output (diversity gaps, contradictory claims, weak evidence) and routes follow-ups to specific agents rather than repeating full pipelines.
*   **Strict Fact Verification & Citation Gating**: Self-corrects claims against retrieved source materials. Only factually verified citations are pushed to memory.
*   **Analytics Dashboard**: Produces detailed summaries of operational performance including token usage, node latencies, and gate approval rates.

---

## 🚀 Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure `langchain`, `langchain_openai`, `langgraph`, `langsmith`, `pinecone`, `requests`, and `python-dotenv` are installed).*

### 2. Configure Environment (`.env`)
Create a `.env` file in the root directory (refer to `.env.example`):
```env
OPENAI_API_KEY=your_openai_or_openrouter_key
PINECONE_API_KEY=your_pinecone_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

# LangSmith Observability
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT="TestAgent"
```

### 3. Run the Systems

*   **To run the Software Engineering Agent**:
    ```bash
    python autonomous_agent.py
    ```
    *Input your developer task (e.g. "Create a library class with an LRU cache, write tests, and verify") and approve files/directories in real-time.*

*   **To run the Deep Research Agent**:
    ```bash
    python deep_research_agent.py "FDA regulation of AI medical devices"
    ```

---

## 📈 Tracing & Observability
Both agents are fully integrated with **LangSmith**. Node performance, LLM call structures, tool observation history, and agent pathing are beautifully recorded in real-time for perfect operational transparency.
