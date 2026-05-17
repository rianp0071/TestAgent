"""
Deep Research Agent v5 -- LangGraph Multi-Agent Pipeline
========================================================
Architecture:
  LangGraph StateGraph with centralized state, specialized research
  agents (gov, academic, community, industry), parallel fan-out,
  adaptive reflection with conditional re-routing, Pinecone vector
  memory for cross-session intelligence, and metrics tracking.

Phase 1: Foundation -- LangGraph StateGraph skeleton.
Phase 2: Specialized research agents with per-agent prompts.
Phase 3: Parallel fan-out via ThreadPoolExecutor.
Phase 4: Enhanced reflection with conditional agent re-routing.
Phase 5: Pinecone vector DB for persistent cross-session memory.

Key v5 improvements over v4:
  1. Centralized ResearchState (TypedDict + reducers).
  2. 4 specialized researcher agents replace monolithic search+extract.
  3. Parallel execution of research agents.
  4. Adaptive reflection routes gaps to specific agents (not all).
  5. Pinecone vector memory for cross-session claim persistence.
  6. Metrics tracked inside state.
"""

import os
import re
import json
import time
import hashlib
import operator
import concurrent.futures
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Annotated, Literal
from typing_extensions import TypedDict
from langsmith import traceable
from openai import OpenAI
import openai

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------
# Metrics Tracker
# ---------------------------------------------
@dataclass
class RunMetrics:
    """Tracks every measurable event across the pipeline."""
    queries_generated: int = 0
    search_results_returned: int = 0
    search_results_after_dedup: int = 0
    claims_extracted: int = 0
    extraction_failures: int = 0
    claims_verified: int = 0
    claims_rejected: int = 0
    verification_failures: int = 0
    claims_passed_gate: int = 0
    claims_used_in_synthesis: int = 0
    final_citation_count: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    total_iterations: int = 0
    total_time_seconds: float = 0.0

    def approval_rate(self) -> str:
        total = self.claims_verified + self.claims_rejected
        if total == 0:
            return "N/A"
        return f"{self.claims_verified / total * 100:.1f}%"

    def print_dashboard(self, agent_stats: dict = None):
        print("\n" + "=" * 60)
        print("  RESEARCH METRICS DASHBOARD")
        print("=" * 60)
        print(f"  Total time              : {self.total_time_seconds:.1f}s")
        print(f"  LLM calls (total/failed): {self.llm_calls} / {self.llm_failures}")
        print(f"  Iterations              : {self.total_iterations}")
        print(f"  --- Search ---")
        print(f"  Queries generated       : {self.queries_generated}")
        print(f"  Results returned (raw)  : {self.search_results_returned}")
        print(f"  Results after dedup     : {self.search_results_after_dedup}")
        print(f"  --- Extraction ---")
        print(f"  Claims extracted        : {self.claims_extracted}")
        print(f"  Extraction failures     : {self.extraction_failures}")
        if agent_stats:
            print(f"  --- Per-Agent Breakdown ---")
            for agent_id, stats in agent_stats.items():
                name = agent_id.replace('_', ' ').title()
                print(f"    {name}: {stats.get('claims_found', 0)} claims, sources: {stats.get('source_types', [])}")
        print(f"  --- Verification ---")
        print(f"  Claims approved         : {self.claims_verified}")
        print(f"  Claims rejected         : {self.claims_rejected}")
        print(f"  Verification failures   : {self.verification_failures}")
        print(f"  Approval rate           : {self.approval_rate()}")
        print(f"  --- Synthesis ---")
        print(f"  Claims passed gate      : {self.claims_passed_gate}")
        print(f"  Claims used in report   : {self.claims_used_in_synthesis}")
        print(f"  Final citation count    : {self.final_citation_count}")
        print("=" * 60)

# Global metrics instance
metrics = RunMetrics()

# ---------------------------------------------
# LLM Client Setup
# ---------------------------------------------
try:
    from firecrawl import FirecrawlApp
except ImportError:
    from firecrawl import Firecrawl as FirecrawlApp

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-OpenRouter-Title": "DeepResearchAgent"
    }
)
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

MODEL = "meta-llama/llama-3.3-70b-instruct"

# ---------------------------------------------
# Pinecone Vector Memory Setup
# ---------------------------------------------
from pinecone import Pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "research-memory"
PINECONE_NAMESPACE = "claims"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    PINECONE_AVAILABLE = True
    print("[Memory] Pinecone vector memory connected.")
except Exception as e:
    PINECONE_AVAILABLE = False
    pinecone_index = None
    print(f"[Memory] Pinecone unavailable ({e}). Running without memory.")

# ---------------------------------------------
# Strict JSON Caller with Fallback Parsing
# ---------------------------------------------
def safe_parse_json(text) -> Optional[dict]:
    """Try multiple strategies to extract JSON from LLM output.
    Handles None, empty strings, and malformed JSON gracefully."""
    if text is None or not isinstance(text, str) or not text.strip():
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2: Extract from ```json ... ``` fences
    try:
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 3: Find first { ... } block
    try:
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return None


@traceable(name="call_llm", run_type="llm")
def call_llm(prompt: str, is_json: bool = False, retries: int = 3) -> Optional[any]:
    """
    Central LLM caller with:
      - rate-limit retry
      - strict JSON mode when supported
      - fallback JSON parsing for malformed responses
      - metrics tracking
      - NEVER crashes the pipeline: returns None on failure
    """
    for attempt in range(retries):
        try:
            metrics.llm_calls += 1
            kwargs = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)

            # Guard: response or content can be None
            if not response or not response.choices:
                metrics.llm_failures += 1
                print(f"  Warning: Empty response from LLM (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None

            raw_text = response.choices[0].message.content

            # Guard: content can be None
            if raw_text is None:
                metrics.llm_failures += 1
                print(f"  Warning: LLM returned None content (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return "" if not is_json else None

            if not is_json:
                return raw_text

            # Strict JSON path
            parsed = safe_parse_json(raw_text)
            if parsed is not None:
                return parsed

            # If strict parse failed, retry with a shorter nudge
            if attempt < retries - 1:
                metrics.llm_failures += 1
                print(f"  Warning: JSON parse failed, retrying ({attempt+1}/{retries})...")
                prompt = prompt + "\n\nIMPORTANT: You MUST return valid JSON only. No extra text."
                continue
            else:
                metrics.llm_failures += 1
                return None

        except openai.RateLimitError:
            if attempt < retries - 1:
                print(f"  Action: Rate limit hit. Sleeping 65s (Attempt {attempt+1}/{retries})...")
                time.sleep(65)
            else:
                metrics.llm_failures += 1
                print(f"  ERROR: Rate limit exceeded after {retries} retries. Skipping this call.")
                return None
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Action: Rate limit hit. Sleeping 65s (Attempt {attempt+1}/{retries})...")
                time.sleep(65)
            else:
                metrics.llm_failures += 1
                print(f"  ERROR: LLM call failed: {type(e).__name__}: {e}")
                return None
    return None


# ---------------------------------------------
# Source Quality Scoring
# ---------------------------------------------
SOURCE_HIERARCHY = {
    'primary_gov': 0,
    'academic': 1,
    'financial_institution': 2,
    'high_quality_secondary': 3,
    'general': 4
}

def get_source_type(url: str) -> str:
    url_lower = url.lower()
    if any(x in url_lower for x in ['.gov', 'sec.gov', 'federalreserve', 'treasury', 'congress.gov', 'gao.gov']):
        return 'primary_gov'
    if '.edu' in url_lower:
        return 'academic'
    if any(x in url_lower for x in ['jpmorgan', 'goldmansachs', 'morganstanley', 'blackrock',
                                     'imf.org', 'worldbank', 'bis.org']):
        return 'financial_institution'
    if any(x in url_lower for x in ['bloomberg', 'reuters', 'wsj', 'ft.com', 'mckinsey',
                                     'economist', 'nytimes', 'washingtonpost']):
        return 'high_quality_secondary'
    return 'general'

def rank_source(item: dict) -> int:
    return SOURCE_HIERARCHY.get(get_source_type(item.get('url', '')), 4)


# =============================================================
# LangGraph Centralized State
# =============================================================
class ResearchState(TypedDict):
    """Centralized state shared across all graph nodes."""
    topic: str
    queries: list                                       # current queries (legacy, used by planner)
    subtasks: list                                       # per-agent subtasks from planner
    search_results: list                                 # results for current iteration
    all_claims: Annotated[list, operator.add]            # accumulated claims across iterations
    gated_claims: list                                   # claims that passed gate
    memory_recalled_claims: list                         # claims recalled from Pinecone
    report: str                                          # final report
    iteration: int                                       # current iteration
    max_iterations: int                                  # iteration cap
    should_stop: bool                                    # reflection decision
    start_time: float                                    # for timing
    agent_stats: dict                                    # per-agent metrics


# =============================================================
# Graph Node Functions
# =============================================================

# =============================================================
# Specialized Agent Configurations
# =============================================================

AGENT_CONFIGS = {
    "gov_researcher": {
        "name": "Government & Policy Researcher",
        "focus": "government reports, regulations, official statistics",
        "query_suffix": "site:.gov OR site:.edu",
        "target_sources": ["SEC", "Federal Reserve", "Treasury", "Congress", "NIH", "CDC"],
        "extraction_prompt": "Only prioritize primary-source institutional evidence. Extract official statistics, regulatory findings, and policy data.",
        "trust_weight": 1.0,
    },
    "academic_researcher": {
        "name": "Academic & Scientific Researcher",
        "focus": "peer-reviewed papers, research findings, university publications",
        "query_suffix": "SSRN OR arxiv OR site:.edu OR journal",
        "target_sources": ["SSRN", "arXiv", "PubMed", "university research"],
        "extraction_prompt": "Extract methodological findings, empirical data, and research conclusions. Focus on quantitative evidence.",
        "trust_weight": 0.9,
    },
    "community_researcher": {
        "name": "Community & Practitioner Researcher",
        "focus": "practitioner insights, real-world experiences, community discussions",
        "query_suffix": "reddit OR forum OR discussion OR blog",
        "target_sources": ["Reddit", "StackExchange", "Hacker News", "practitioner blogs"],
        "extraction_prompt": "Extract practitioner observations and real-world implementation details. NOT authoritative facts — mark confidence as medium or low.",
        "trust_weight": 0.5,
    },
    "industry_researcher": {
        "name": "Industry & Financial Researcher",
        "focus": "industry reports, market analysis, enterprise data",
        "query_suffix": "McKinsey OR Gartner OR Bloomberg OR Reuters OR report",
        "target_sources": ["McKinsey", "Bloomberg", "Reuters", "Gartner", "company reports"],
        "extraction_prompt": "Extract market data, industry trends, company-specific facts, and financial metrics. Focus on enterprise-grade evidence.",
        "trust_weight": 0.8,
    },
}


@traceable(name="planner_node")
def planner_node(state: ResearchState) -> dict:
    """Phase 1: Generate per-agent subtasks from topic."""
    topic = state["topic"]
    iteration = state.get("iteration", 0)

    print("\n--- PHASE 1: PLANNING ---")
    print(f"  Analyzing topic: '{topic}' (iteration {iteration})")

    if iteration == 0:
        prompt = f"""You are an expert research planner. The user wants to learn about: "{topic}".

You have 4 specialized research agents:
1. gov_researcher — searches .gov/.edu sources for official data
2. academic_researcher — searches academic papers (SSRN, arXiv, PubMed)
3. community_researcher — searches Reddit, forums, practitioner blogs
4. industry_researcher — searches McKinsey, Bloomberg, Reuters, Gartner

Generate 1-2 targeted search queries for EACH agent. Each query should be
optimized for that agent's source domain.

Return ONLY a JSON object:
{{
    "plan": "Brief research plan",
    "subtasks": [
        {{"agent": "gov_researcher", "queries": ["query1"]}},
        {{"agent": "academic_researcher", "queries": ["query1"]}},
        {{"agent": "community_researcher", "queries": ["query1"]}},
        {{"agent": "industry_researcher", "queries": ["query1"]}}
    ]
}}"""

        result = call_llm(prompt, is_json=True)
        if not result or "subtasks" not in result:
            print("  ERROR: Planning failed. Using fallback subtasks.")
            subtasks = [
                {"agent": "gov_researcher", "queries": [f"{topic} site:.gov"]},
                {"agent": "academic_researcher", "queries": [f"{topic} research paper"]},
                {"agent": "community_researcher", "queries": [f"{topic} reddit discussion"]},
                {"agent": "industry_researcher", "queries": [f"{topic} industry report"]},
            ]
        else:
            subtasks = result["subtasks"]
            print(f"  Plan: {result.get('plan', 'N/A')}")

        total_queries = sum(len(s.get("queries", [])) for s in subtasks)
        metrics.queries_generated += total_queries
        print(f"  Generated {total_queries} queries across {len(subtasks)} agents")
        for s in subtasks:
            print(f"    -> {s['agent']}: {s.get('queries', [])}")
        return {"subtasks": subtasks, "iteration": 1, "start_time": time.time()}
    else:
        # Follow-up: targeted subtasks come directly from reflection_node
        targeted_subtasks = state.get("subtasks", [])
        if targeted_subtasks:
            print(f"  Using {len(targeted_subtasks)} targeted agent subtasks from reflection.")
            for s in targeted_subtasks:
                print(f"    -> {s['agent']}: {s.get('queries', [])}")
            return {"iteration": iteration + 1}
        # Fallback: if reflection only gave queries, distribute round-robin
        queries = state.get("queries", [])
        print(f"  Using {len(queries)} follow-up queries (round-robin distribution).")
        agents = list(AGENT_CONFIGS.keys())
        subtasks = [{"agent": a, "queries": []} for a in agents]
        for i, q in enumerate(queries):
            subtasks[i % len(agents)]["queries"].append(q)
        subtasks = [s for s in subtasks if s["queries"]]
        return {"subtasks": subtasks, "iteration": iteration + 1}


# =============================================================
# Shared Research Agent Logic (used by all 4 specialized agents)
# =============================================================

class AgentSubgraphState(TypedDict):
    topic: str
    agent_id: str
    queries: list
    unique_results: list
    claims: list

@traceable(name="agent_search_node")
def agent_search_node(state: AgentSubgraphState) -> dict:
    agent_id = state["agent_id"]
    topic = state["topic"]
    queries = state["queries"]
    config = AGENT_CONFIGS[agent_id]
    agent_name = config["name"]

    print(f"\n  [{agent_name}] Searching with {len(queries)} queries...")
    results = []
    def fetch_query(q):
        try:
            search_result = app.search(query=q)
            data = getattr(search_result, 'web', [])
            extracted = []
            for item in data:
                url = getattr(item, 'url', 'Unknown URL')
                content = getattr(item, 'description', '') or getattr(item, 'title', 'No content')
                if content and len(content.strip()) > 20:
                    extracted.append({"query": q, "url": url, "content": content, "agent": agent_id})
            return extracted
        except Exception as e:
            print(f"    Error searching '{q}': {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_query, q): q for q in queries[:3]}
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    metrics.search_results_returned += len(results)

    seen_urls = set()
    unique = []
    for r in results:
        if r['url'] not in seen_urls:
            unique.append(r)
            seen_urls.add(r['url'])

    metrics.search_results_after_dedup += len(unique)
    print(f"    [{agent_name}] {len(unique)} unique results found")
    return {"unique_results": unique}

@traceable(name="agent_extract_node")
def agent_extract_node(state: AgentSubgraphState) -> dict:
    agent_id = state["agent_id"]
    topic = state["topic"]
    unique_results = state.get("unique_results", [])
    if not unique_results:
        return {"claims": []}

    config = AGENT_CONFIGS[agent_id]
    agent_name = config["name"]
    extraction_hint = config["extraction_prompt"]

    new_claims = []
    top_results = unique_results[:10]
    batches = [top_results[i:i + 5] for i in range(0, len(top_results), 5)]

    for batch_idx, batch in enumerate(batches):
        batch_text = ""
        for idx, res in enumerate(batch):
            batch_text += f"\n[Source {idx}] URL: {res['url']}\nText: {res['content']}\n"

        prompt = f"""Extract concrete facts from these search results about "{topic}".

YOU ARE: {agent_name}
FOCUS: {extraction_hint}

RULES:
- Only extract specific, verifiable claims (data points, dates, names, amounts).
- Skip vague or opinion-based statements.
- Each claim MUST include a direct quote from the source text.

Sources:
{batch_text}

Return ONLY a JSON object:
{{
    "claims": [
        {{
            "source_id": 0,
            "topic_tag": "short tag",
            "claim": "specific factual claim",
            "confidence": "high" or "medium" or "low",
            "quote": "exact quote from text"
        }}
    ]
}}"""
        try:
            extracted = call_llm(prompt, is_json=True)
            if not extracted or "claims" not in extracted:
                metrics.extraction_failures += 1
                continue

            claims = extracted["claims"]
            for c in claims:
                sid = c.get("source_id")
                if not isinstance(sid, int) or sid < 0 or sid >= len(batch):
                    continue
                if not c.get("claim") or not c.get("quote"):
                    continue

                url = batch[sid]['url']
                c["source_url"] = url
                c["source_type"] = get_source_type(url)
                c["id"] = hashlib.md5(f"{url}-{c['claim']}".encode()).hexdigest()[:8]
                c["verified"] = False
                c["gate_passed"] = False
                c["agent"] = agent_id
                new_claims.append(c)

        except Exception as e:
            metrics.extraction_failures += 1
            print(f"  Batch {batch_idx+1} failed: {e}")

    metrics.claims_extracted += len(new_claims)
    print(f"    [{agent_name}] Extracted {len(new_claims)} claims")
    return {"claims": new_claims}

def build_agent_subgraph() -> StateGraph:
    builder = StateGraph(AgentSubgraphState)
    builder.add_node("agent_search", agent_search_node)
    builder.add_node("agent_extract", agent_extract_node)
    builder.add_edge(START, "agent_search")
    builder.add_edge("agent_search", "agent_extract")
    builder.add_edge("agent_extract", END)
    return builder.compile()

# =============================================================
# Parallel Research Dispatcher
# =============================================================

@traceable(name="research_dispatcher")
def research_dispatcher(state: ResearchState) -> dict:
    """Fan-out: run all specialized agent subgraphs in parallel."""
    topic = state["topic"]
    subtasks = state.get("subtasks", [])

    print("\n--- PHASE 2+3: PARALLEL AGENT SUBGRAPHS ---")
    print(f"  Dispatching {len(subtasks)} agent subgraphs in parallel...")

    all_new_claims = []
    agent_stats = {}
    
    agent_app = build_agent_subgraph()

    def run_agent_subgraph(subtask):
        agent_id = subtask.get("agent", "gov_researcher")
        queries = subtask.get("queries", [])
        if not queries:
            return agent_id, []
        
        initial_sub_state = {
            "topic": topic,
            "agent_id": agent_id,
            "queries": queries,
            "unique_results": [],
            "claims": []
        }
        # Invoke the LangGraph subgraph
        final_sub_state = agent_app.invoke(initial_sub_state)
        return agent_id, final_sub_state.get("claims", [])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_agent_subgraph, st) for st in subtasks]
        for future in concurrent.futures.as_completed(futures):
            agent_id, claims = future.result()
            all_new_claims.extend(claims)
            agent_stats[agent_id] = {
                "claims_found": len(claims),
                "source_types": list(set(c.get("source_type", "general") for c in claims)),
            }

    # Sort by source quality
    all_new_claims.sort(key=lambda c: SOURCE_HIERARCHY.get(c.get("source_type", "general"), 4))

    print(f"\n  --- Agent Summary ---")
    for agent_id, stats in agent_stats.items():
        name = AGENT_CONFIGS.get(agent_id, {}).get("name", agent_id)
        print(f"    {name}: {stats['claims_found']} claims, sources: {stats['source_types']}")
    print(f"  Total new claims: {len(all_new_claims)}")

    return {"all_claims": all_new_claims, "agent_stats": agent_stats}


@traceable(name="verify_node")
def verify_node(state: ResearchState) -> dict:
    """Phase 4: Verify claims against their supporting quotes."""
    topic = state["topic"]
    all_claims = state.get("all_claims", [])
    print("\n--- PHASE 4: VERIFICATION ---")

    unverified = [c for c in all_claims if not c.get('verified', False)]
    if not unverified:
        print("  All claims already verified.")
        return {}

    batches = [unverified[i:i + 12] for i in range(0, len(unverified), 12)]

    for batch in batches:
        batch_input = json.dumps([{
            "id": c["id"],
            "claim": c["claim"],
            "quote": c["quote"],
            "source_type": c["source_type"]
        } for c in batch], indent=2)

        prompt = f"""You are a strict fact-checker for the topic "{topic}".

For each claim below, determine if the "quote" genuinely supports the "claim".
Reject claims that are:
- Too vague or generic to be useful
- Not actually supported by the quote
- Duplicates of other claims in the batch

Claims:
{batch_input}

Return ONLY a JSON object:
{{
    "verifications": [
        {{
            "id": "claim_id",
            "status": "approved" or "rejected",
            "reason": "one sentence explanation"
        }}
    ]
}}"""

        try:
            result = call_llm(prompt, is_json=True)
            if not result or "verifications" not in result:
                metrics.verification_failures += 1
                continue

            vmap = {v["id"]: v for v in result["verifications"] if "id" in v}
            for c in batch:
                v = vmap.get(c["id"])
                if v and v.get("status", "").lower() == "approved":
                    c["verified"] = True
                    c["verification_reason"] = v.get("reason", "")
                    metrics.claims_verified += 1
                else:
                    c["verified"] = False
                    c["verification_reason"] = v.get("reason", "No match") if v else "Missing from verifier output"
                    metrics.claims_rejected += 1

        except Exception as e:
            metrics.verification_failures += 1
            print(f"  Verification batch failed: {e}")

    approved = sum(1 for c in all_claims if c.get("verified"))
    print(f"  Result: {approved} approved / {len(all_claims)} total")

    # NOTE: We replace all_claims entirely (not append) since we mutated in place.
    # Use a special return to signal full replacement.
    return {}


@traceable(name="citation_gate_node")
def citation_gate_node(state: ResearchState) -> dict:
    """Phase 5: Filter to only well-sourced, verified claims."""
    all_claims = state.get("all_claims", [])
    print("\n--- PHASE 5: CITATION GATE ---")

    gated = []
    for c in all_claims:
        if not c.get("verified", False):
            continue
        url = c.get("source_url", "")
        if not url or url == "Unknown URL":
            continue
        if not c.get("quote", "").strip():
            continue
        stype = c.get("source_type", "general")
        conf = c.get("confidence", "low").lower()
        if stype == "general" and conf == "low":
            continue
        c["gate_passed"] = True
        gated.append(c)

    metrics.claims_passed_gate = len(gated)
    metrics.total_iterations = state.get("iteration", 1)
    print(f"  {len(gated)} claims passed the citation gate (out of {len(all_claims)} total).")
    return {"gated_claims": gated}


@traceable(name="reflection_node")
def reflection_node(state: ResearchState) -> dict:
    """Phase 6: Adaptive reflection with targeted agent re-routing.
    
    Analyzes:
      - Per-agent claim counts and performance
      - Source type diversity gaps
      - Confidence distribution
      - Missing subtopics
      - Contradictory or weak claims
    
    Returns targeted subtasks routed to specific agents, NOT generic queries.
    """
    topic = state["topic"]
    gated_claims = state.get("gated_claims", [])
    all_claims = state.get("all_claims", [])
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 2)
    agent_stats = state.get("agent_stats", {})

    print("\n--- PHASE 6: ADAPTIVE REFLECTION ---")

    # Hard stop at max iterations
    if iteration >= max_iterations:
        print(f"  Reached max iterations ({max_iterations}). Stopping.")
        return {"should_stop": True}

    # ---- Build rich analysis for the LLM ----

    # Per-agent claim breakdown
    agent_claim_counts = {}
    agent_source_types = {}
    agent_confidence = {}
    for c in all_claims:
        agent = c.get("agent", "unknown")
        agent_claim_counts[agent] = agent_claim_counts.get(agent, 0) + 1
        agent_source_types.setdefault(agent, set()).add(c.get("source_type", "general"))
        conf = c.get("confidence", "low").lower()
        agent_confidence.setdefault(agent, {"high": 0, "medium": 0, "low": 0})
        if conf in agent_confidence[agent]:
            agent_confidence[agent][conf] += 1

    # Subtopic coverage
    tag_counts = {}
    source_types_used = set()
    for c in gated_claims:
        tag = c.get("topic_tag", "unknown")
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        source_types_used.add(c.get("source_type", "general"))

    # Identify underperforming agents (0 gated claims)
    gated_by_agent = {}
    for c in gated_claims:
        a = c.get("agent", "unknown")
        gated_by_agent[a] = gated_by_agent.get(a, 0) + 1

    underperforming = [
        a for a in AGENT_CONFIGS.keys()
        if gated_by_agent.get(a, 0) == 0
    ]

    analysis = json.dumps({
        "total_verified_claims": len(gated_claims),
        "subtopic_coverage": tag_counts,
        "source_types_present": list(source_types_used),
        "per_agent_gated_claims": gated_by_agent,
        "per_agent_confidence": {k: dict(v) if isinstance(v, dict) else v for k, v in agent_confidence.items()},
        "underperforming_agents": underperforming,
        "sample_claims": [c["claim"] for c in gated_claims[:8]],
        "available_agents": {
            aid: {"name": cfg["name"], "focus": cfg["focus"]}
            for aid, cfg in AGENT_CONFIGS.items()
        },
    }, indent=2)

    prompt = f"""You are an expert research supervisor reviewing evidence for: "{topic}".

Current evidence analysis:
{analysis}

Your job is to identify SPECIFIC gaps and route follow-up research to the RIGHT agent.

Analyze:
1. Which subtopics are MISSING entirely?
2. Which claims have WEAK evidence (only general sources, low confidence)?
3. Are there CONTRADICTIONS between claims that need resolution?
4. Which agents UNDERPERFORMED and should be retried with better queries?
5. Is source diversity sufficient (gov, academic, industry, community)?

For each gap, assign it to the BEST agent and write a targeted search query.

Return ONLY a JSON object:
{{
    "analysis": "Brief assessment of current evidence quality",
    "missing_subtopics": ["subtopic1", "subtopic2"],
    "weak_areas": ["area needing stronger sources"],
    "contradictions": ["any contradictory claims found"],
    "stop": true or false,
    "targeted_subtasks": [
        {{"agent": "gov_researcher", "queries": ["specific query for this gap"], "reason": "why this agent"}},
        {{"agent": "academic_researcher", "queries": ["specific query"], "reason": "why"}}
    ]
}}"""

    result = call_llm(prompt, is_json=True)
    if not result:
        print("  Reflection failed, stopping to avoid infinite loop.")
        return {"should_stop": True}

    print(f"  Analysis: {result.get('analysis', 'N/A')}")
    print(f"  Missing subtopics: {result.get('missing_subtopics', [])}")
    print(f"  Weak areas: {result.get('weak_areas', [])}")
    print(f"  Contradictions: {result.get('contradictions', [])}")
    if underperforming:
        print(f"  Underperforming agents: {underperforming}")

    if result.get("stop", False):
        print("  Decision: STOP -- evidence is sufficient.")
        return {"should_stop": True}

    targeted = result.get("targeted_subtasks", [])
    if not targeted:
        print("  Decision: STOP -- no targeted subtasks generated.")
        return {"should_stop": True}

    # Validate agent IDs and filter invalid ones
    valid_agents = set(AGENT_CONFIGS.keys())
    valid_subtasks = []
    for st in targeted:
        agent_id = st.get("agent", "")
        queries = st.get("queries", [])
        if agent_id in valid_agents and queries:
            valid_subtasks.append({"agent": agent_id, "queries": queries})
            reason = st.get("reason", "")
            print(f"    -> {agent_id}: {queries} ({reason})")

    if not valid_subtasks:
        print("  Decision: STOP -- no valid targeted subtasks after validation.")
        return {"should_stop": True}

    total_queries = sum(len(s["queries"]) for s in valid_subtasks)
    metrics.queries_generated += total_queries
    print(f"  Decision: CONTINUE -- {len(valid_subtasks)} agents targeted with {total_queries} queries.")
    return {"should_stop": False, "subtasks": valid_subtasks}


@traceable(name="synthesis_node")
def synthesis_node(state: ResearchState) -> dict:
    """Phase 7: Generate final report from gated claims + memory."""
    topic = state["topic"]
    gated_claims = state.get("gated_claims", [])
    memory_claims = state.get("memory_recalled_claims", [])
    print("\n--- PHASE 7: SYNTHESIS ---")

    # Merge current claims with recalled memory claims (deduped by id)
    seen_ids = set()
    combined = []
    for c in gated_claims:
        cid = c.get("id", "")
        if cid not in seen_ids:
            combined.append(c)
            seen_ids.add(cid)
    memory_used = 0
    for c in memory_claims:
        cid = c.get("id", "")
        if cid and cid not in seen_ids:
            combined.append(c)
            seen_ids.add(cid)
            memory_used += 1

    if memory_used > 0:
        print(f"  Merged {memory_used} claims from prior research memory.")

    def score(c):
        src_rank = SOURCE_HIERARCHY.get(c.get("source_type", "general"), 4)
        conf_rank = {"high": 0, "medium": 1, "low": 2}.get(c.get("confidence", "low").lower(), 2)
        return (src_rank, conf_rank)

    combined.sort(key=score)
    top = combined[:30]
    metrics.claims_used_in_synthesis = len(top)

    claims_json = json.dumps([{
        "id": c["id"],
        "tag": c.get("topic_tag", ""),
        "claim": c.get("claim", ""),
        "url": c.get("source_url", ""),
        "source": "prior_memory" if c.get("from_memory") else "current_research"
    } for c in top], indent=2)

    prompt = f"""You are an expert researcher writing a final report on: "{topic}".

RULES:
1. Use ONLY the verified claims below. Do NOT add outside knowledge.
2. If a fact is not in the claims list, do NOT mention it.
3. Every factual sentence MUST end with a citation like [URL].
4. Organize your report around the evidence, not around general knowledge.

Process:
Step 1 — Write an outline of 3-5 sections based on the claim tags.
Step 2 — Map each claim ID to the section it belongs to.
Step 3 — Write the full report with inline [URL] citations.

Verified Claims:
{claims_json}

Format:
## Outline
- Section 1: ...
- Section 2: ...

## Claim Mapping
- Section 1: [id1, id2]
- Section 2: [id3, id4]

## Final Report
[Full prose with [URL] citations after every claim]"""

    report = call_llm(prompt, is_json=False)
    if not report:
        report = "ERROR: Synthesis failed. Please retry."

    citation_count = len(re.findall(r'\[https?://[^\]]+\]', report))
    metrics.final_citation_count = citation_count

    elapsed = time.time() - state.get("start_time", time.time())
    metrics.total_time_seconds = elapsed

    return {"report": report}


# =============================================================
# Pinecone Memory Nodes
# =============================================================

@traceable(name="memory_recall_node")
def memory_recall_node(state: ResearchState) -> dict:
    """Query Pinecone for prior claims relevant to this topic.
    Runs once at the start, before research begins.
    Recalled claims are injected into the synthesis context.
    """
    topic = state["topic"]
    print("\n--- MEMORY RECALL ---")

    if not PINECONE_AVAILABLE or pinecone_index is None:
        print("  Pinecone not available. Skipping memory recall.")
        return {"memory_recalled_claims": []}

    try:
        results = pinecone_index.search(
            namespace=PINECONE_NAMESPACE,
            top_k=15,
            inputs={"text": topic},
        )

        recalled = []
        # SearchRecordsResponse has .result.hits
        result_obj = getattr(results, 'result', None)
        hits = getattr(result_obj, 'hits', []) if result_obj else []
        for hit in hits:
            # Pinecone v9 Hit uses .score and .id (not _score, _id)
            score = getattr(hit, 'score', None) or getattr(hit, '_score', None) or 0
            if score < 0.03:
                continue
            fields = getattr(hit, 'fields', {}) or {}
            hit_id = getattr(hit, 'id', None) or getattr(hit, '_id', '') or ''
            recalled.append({
                "id": hit_id,
                "claim": fields.get("claim_text", ""),
                "source_url": fields.get("source_url", ""),
                "source_type": fields.get("source_type", ""),
                "topic_tag": fields.get("topic_tag", ""),
                "agent": fields.get("agent", "memory"),
                "confidence": fields.get("confidence", "medium"),
                "verified": True,
                "gate_passed": True,
                "from_memory": True,
                "relevance_score": score,
            })

        print(f"  Recalled {len(recalled)} relevant claims from prior research.")
        for c in recalled[:5]:
            print(f"    [{c.get('source_type', '?')}] {c['claim'][:80]}... (score: {c['relevance_score']:.3f})")

        return {"memory_recalled_claims": recalled}

    except Exception as e:
        print(f"  Memory recall error: {e}")
        return {"memory_recalled_claims": []}


@traceable(name="memory_store_node")
def memory_store_node(state: ResearchState) -> dict:
    """Store verified, gated claims in Pinecone for future recall.
    Runs after citation gate — only stores high-quality claims.
    """
    gated_claims = state.get("gated_claims", [])
    topic = state["topic"]
    print("\n--- MEMORY STORE ---")

    if not PINECONE_AVAILABLE or pinecone_index is None:
        print("  Pinecone not available. Skipping memory store.")
        return {}

    # Filter to only new claims (not already from memory)
    new_claims = [c for c in gated_claims if not c.get("from_memory", False)]
    if not new_claims:
        print("  No new claims to store.")
        return {}

    records = []
    for c in new_claims:
        record = {
            "_id": c.get("id", hashlib.md5(c["claim"].encode()).hexdigest()[:12]),
            "claim_text": c.get("claim", ""),
            "source_url": c.get("source_url", ""),
            "source_type": c.get("source_type", "general"),
            "topic_tag": c.get("topic_tag", ""),
            "topic": topic,
            "agent": c.get("agent", "unknown"),
            "confidence": c.get("confidence", "medium"),
            "verified": True,
        }
        records.append(record)

    try:
        # Upsert in batches of 50
        for i in range(0, len(records), 50):
            batch = records[i:i + 50]
            pinecone_index.upsert_records(namespace=PINECONE_NAMESPACE, records=batch)

        print(f"  Stored {len(records)} verified claims in Pinecone for future recall.")
        return {}
    except Exception as e:
        print(f"  Memory store error: {e}")
        return {}


# =============================================================
# Conditional Edge: Should we loop back or synthesize?
# =============================================================

def should_continue(state: ResearchState) -> Literal["planner_node", "synthesis_node"]:
    """Route based on reflection decision."""
    if state.get("should_stop", True):
        return "synthesis_node"
    return "planner_node"


# =============================================================
# Build the LangGraph
# =============================================================

def build_research_graph() -> StateGraph:
    """Construct and compile the research agent graph."""
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("memory_recall_node", memory_recall_node)
    builder.add_node("planner_node", planner_node)
    builder.add_node("research_dispatcher", research_dispatcher)
    builder.add_node("verify_node", verify_node)
    builder.add_node("citation_gate_node", citation_gate_node)
    builder.add_node("memory_store_node", memory_store_node)
    builder.add_node("reflection_node", reflection_node)
    builder.add_node("synthesis_node", synthesis_node)

    # Edges: memory recall -> planner -> research -> verify -> gate -> store -> reflect
    builder.add_edge(START, "memory_recall_node")
    builder.add_edge("memory_recall_node", "planner_node")
    builder.add_edge("planner_node", "research_dispatcher")
    builder.add_edge("research_dispatcher", "verify_node")
    builder.add_edge("verify_node", "citation_gate_node")
    builder.add_edge("citation_gate_node", "memory_store_node")
    builder.add_edge("memory_store_node", "reflection_node")

    # Conditional: loop back to planner or go to synthesis
    builder.add_conditional_edges(
        "reflection_node",
        should_continue,
        ["planner_node", "synthesis_node"]
    )

    builder.add_edge("synthesis_node", END)

    return builder.compile()


# =============================================================
# Main Entry Point
# =============================================================

def run_deep_research(topic: str, max_iterations: int = 2):
    """Run the LangGraph-based research pipeline."""
    global metrics
    metrics = RunMetrics()

    print("=" * 60)
    print("  DEEP RESEARCH AGENT v5 — LangGraph Pipeline")
    print(f"  Topic: {topic}")
    print("=" * 60)

    graph = build_research_graph()

    # Invoke the graph with initial state
    initial_state = {
        "topic": topic,
        "queries": [],
        "subtasks": [],
        "search_results": [],
        "all_claims": [],
        "gated_claims": [],
        "memory_recalled_claims": [],
        "report": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_stop": False,
        "start_time": time.time(),
        "agent_stats": {},
    }

    final_state = graph.invoke(initial_state)

    report = final_state.get("report", "No report generated.")

    print("\n" + "=" * 60)
    print("  FINAL RESEARCH REPORT")
    print("=" * 60)
    print(report)

    # Print metrics dashboard with per-agent stats
    agent_stats = final_state.get("agent_stats", {})
    metrics.print_dashboard(agent_stats=agent_stats)

    return report


# ---------------------------------------------
# Entry Point
# ---------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        topic = input("  Enter your research topic: ")
        print("=" * 60)

    if not topic.strip():
        print("Error: No topic provided.")
    else:
        run_deep_research(topic)
