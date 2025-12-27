# Autonomous MRD Agent — Prompt → Synthesis → Output

This document outlines the end-to-end flow your system follows from receiving a prompt through planning, research, synthesis, validation, and delivery, including production-grade branches and safeguards. Use this as the content for a Figma/ FigJam diagram and as reference notes.

## System Components (repo references)
- Entry point: [main.py](main.py)
- Orchestration: [orchestration/state_machine.py](orchestration/state_machine.py)
- Handlers: [handlers/planning.py](handlers/planning.py), [handlers/synthesis.py](handlers/synthesis.py)
- Agents: [agents/research_agents.py](agents/research_agents.py)
- Adapters: [adapters/gemini_client.py](adapters/gemini_client.py), [adapters/research_generator.py](adapters/research_generator.py)
- Models: [models/mrd.py](models/mrd.py), [models/core.py](models/core.py)
- Output: [output/](output/)

## High-Level Flow
1. Prompt Received
   - Validate input: required fields, length limits, allowed characters
   - Normalize: trim, standardize language/domain hints, extract constraints
   - Branches:
     - Invalid → return error + corrective guidance
     - Valid → advance to PLANNING

2. Planning (INIT → PLANNING in state machine)
   - Derive objectives, tasks, constraints, acceptance criteria
   - Decide tools & data needed (web research, internal knowledge, prior outputs)
   - Branches:
     - Data needed → Research Agents
     - No extra data → proceed directly to Synthesis

3. Research & Data Acquisition
   - Select tools (e.g., Gemini, internal APIs), construct queries
   - Fetch data, paginate results, handle rate limits
   - Aggregate: deduplicate, cluster by topic, rank sources by quality and recency
   - Branches:
     - API OK → proceed
     - Rate-limited/Failed → retry with exponential backoff, apply circuit breaker; fall back to cache or reduced scope
     - Insufficient evidence → loop back to Planning to refine queries/tasks

4. Synthesis
   - Outline: section structure, key claims, citations plan
   - Draft: generate content grounded in sources; insert citations
   - Branches:
     - If evidence gaps emerge → back to Planning/Research for more data

5. Validation Gates
   - Structure: schema checks (Markdown, JSON), heading/order consistency
   - Safety: PII removal, toxicity filters, policy compliance
   - Factuality: cross-source verification for claims; flag hallucinations
   - Branches:
     - Any check fails → return to Planning to fix inputs or constraints
     - All pass → proceed to formatting

6. Formatting & Output Preparation
   - Produce Markdown and JSON artifacts
   - Apply naming convention with stable IDs and timestamps (see samples in [output/](output/))
   - Branches:
     - Missing fields → correct & re-validate

7. Delivery & Persistence
   - Return via API/CLI
   - Persist to filesystem (Markdown & JSON)
   - Emit logs, metrics, traces for observability
   - Branches:
     - Notify downstream systems (events)
     - Async job scheduling for long-running tasks

## Production-Grade Branches & Safeguards
- Authentication & Authorization: gate tools and outputs by user/role
- Feature Flags: toggle new tools/flows, staged rollouts
- Rate Limiting & Quotas: per-user and per-integration
- Retries & Circuit Breakers: protect upstream APIs and keep latency bounded
- Caching: memoize recent research results and common queries
- Timeouts & Backpressure: cancel or defer long operations; queue work
- Observability: structured logs, metrics (latency, success rate), tracing through state transitions
- Data Governance: redact sensitive content; audit trails for outputs
- Disaster Recovery: write-ahead logs; periodic snapshots of key artifacts
- Cost Controls: cap external API spend; degrade gracefully under budget pressure

## Mermaid Flowchart (paste into FigJam generator if needed)
```mermaid
flowchart TD
    subgraph Input Layer
        UP[User Prompt] --> PI[Prompt Interpreter]
        PI --> RP[Research Plan Generator]
    end

    subgraph Orchestration Layer - State Machine
        RP --> |Plan| HITL1{Human Approval?}
        HITL1 --> |Approved| SM[State Manager]
        HITL1 --> |Rejected| RP
        
        SM --> |RESEARCH| RO[Research Orchestrator]
        SM --> |SYNTHESIS| SO[Synthesis Orchestrator]
        SM --> |VALIDATION| VO[Validation Orchestrator]
        SM --> |HUMAN_REVIEW| HR{Human Review?}
        SM --> |COMPLETE| OUT[Output Handler]
    end

    subgraph Research Agents Pool
        RO --> MA[Market Analysis Agent]
        RO --> CA[Competitor Analysis Agent]
        RO --> SA[Sentiment Analysis Agent]
        RO --> RA[Regulatory Analysis Agent]
        
        MA --> TC{Tool Coordinator}
        CA --> TC
        SA --> TC
        RA --> TC
    end

    subgraph Tool Layer - Mocked Interfaces
        TC --> ST[search_sensor_tower]
        TC --> AS[analyze_sentiment]
        TC --> CR[check_regulatory_compliance]
        TC --> WS[web_search]
        TC --> SC[social_scraper]
    end

    subgraph Validation & Error Handling (Research Data)
        TC --> |Results| DV[Data Validator]
        DV --> |Valid| SM
        DV --> |Invalid/Empty| RH[Retry Handler]
        RH --> |Retry| TC
        RH --> |Max Retries| FB[Fallback Strategy]
        FB --> SM
    end

    subgraph Synthesis Layer
        SO --> MG[MRD Generator]
        MG --> SV[Schema Validator (structure)]
        SV --> |Valid| SM
        SV --> |Invalid| MG
    end

    %% MRD validation and review flow
    VO --> |Approved| SM
    VO --> |Revise| SO
    VO --> |Needs Review| HR
    HR --> |Approved| SM
    HR --> |Revisions| SO

    subgraph Output
        OUT --> JSON[(Structured JSON)]
        OUT --> DB[(Database)]
        OUT --> API[API Response]
    end

    style SM fill:#e1f5fe
    style HITL1 fill:#fff3e0
    style HR fill:#fff3e0
    style DV fill:#e8f5e9
```


## How to Use
- Open the FigJam link (below) and edit the diagram as needed.
- Keep repo references attached to nodes for faster dev handoffs.
- Use the Production-Grade list to annotate branches in Figma with callouts.

## Optional Notes
- Align node names with actual state names in [orchestration/state_machine.py](orchestration/state_machine.py).
- If you add new tools or handlers, extend the flow with new branches and gates.

## Research Agents — Function & Data Access
This section details how research agents operate, obtain data, and return results in a production setup.

### Roles & Responsibilities
- **Planner → Agents**: The planner identifies information gaps, defines queries, and delegates to research agents.
- **Research Agents**: Execute queries via adapters, fetch and normalize data, score evidence, and package results.
- **Adapters**: Encapsulate external APIs (e.g., Gemini in [adapters/gemini_client.py](adapters/gemini_client.py)) and internal data sources.

### Tool Selection & Querying
- **Selection**: Based on domain tags, confidence scores, and cost caps, choose among web search, LLM retrieval, or internal indexes.
- **Query Construction**: Use structured prompts from planner constraints (topic, scope, date bounds, regions). Include pagination hints and language preferences.
- **Authentication**: Load API keys from environment/secret store; never embed in code.
- **Cost Controls**: Enforce per-request and per-run budgets; degrade to cached or narrower queries when limits approach.

### Data Retrieval & Resilience
- **Rate Limits**: Track per-tool quotas; apply exponential backoff + jitter, then circuit-break on sustained failures.
- **Timeouts**: Bound request time; fall back to cached summaries or previously ranked sources.
- **Caching**: Memoize frequent queries (keyed by normalized query + constraints) with TTL; invalidate on domain changes.
- **Observability**: Emit structured logs with query, latency, status; record metrics per adapter and agent.

### Normalization & Evidence Packaging
- **Deduplication**: Use URL canonicalization, content hashes, and title similarity to merge duplicates.
- **Ranking**: Score by source credibility, freshness, topical relevance, and consistency across multiple sources.
- **Extraction**: Capture key claims, snippets, and metadata (author, date, outlet) for citation.
- **Return Contract**: Agents return a typed `ResearchBundle` comprising the query, parameters, and an array of `Evidence` items with scores and citations.

### Orchestration Integration
- **State Transitions**: On success, the state machine advances from PLANNING → RESEARCH → SYNTHESIS (see [orchestration/state_machine.py](orchestration/state_machine.py)).
- **Feedback Loops**: If `Evidence` density or confidence is below threshold, planner refines queries and re-invokes agents.
- **Error Paths**: Persistent adapter failures short-circuit to a reduced-scope synthesis with disclaimers, logged for later re-run.

## Pydantic AI — Validation & Structured Outputs
Validation is ensured by enforcing typed models throughout the pipeline and using Pydantic AI to drive LLM steps that must emit structured data.

### Typed Schemas
- **Inputs**: `PromptSpec` defines fields like `topic`, `constraints`, `max_tokens`, `domain`, with strict types and bounds.
- **Research**: `Evidence` and `ResearchBundle` capture source-level details and aggregate results.
- **Synthesis**: `MRDOutput` contains sections, claims, and `Citation` arrays, plus a machine-readable JSON alongside Markdown.

### Validation Strategy
- **Model Validation**: All inbound/outbound artifacts pass `model_validate()` before progressing.
- **Field Validators**: Enforce non-empty strings, URL formats, date ranges, and minimum evidence counts.
- **Cross-Checks**: Match citations to claims; verify section headings follow schema and required sections exist.
- **LLM Output Parsing**: Pydantic AI runs specify `response_model`, ensuring the LLM must produce JSON that conforms to the schema.
- **Autofix Loop**: On validation failure, the system returns to the nearest stage (planning or synthesis) with a corrective instruction.

### Example — Agent with Pydantic AI
```python
from pydantic import BaseModel, HttpUrl, Field
from pydantic_ai import Agent, RunOptions

class Evidence(BaseModel):
  url: HttpUrl
  title: str = Field(min_length=5)
  snippet: str = Field(min_length=20)
  outlet: str
  published_at: str  # ISO date string
  quality_score: float = Field(ge=0, le=1)

class ResearchBundle(BaseModel):
  query: str
  constraints: dict
  evidences: list[Evidence]

gemini = Agent(
  model="gemini-1.5-pro",
  system_prompt=(
    "Return high-quality, cited evidence as JSON matching ResearchBundle. "
    "Prefer primary sources; include publication date and outlet."
  ),
)

def run_research(query: str, constraints: dict) -> ResearchBundle:
  result = gemini.run(
    {"query": query, "constraints": constraints},
    options=RunOptions(response_model=ResearchBundle)
  )
  # Pydantic AI ensures structured output; we still validate explicitly.
  bundle = ResearchBundle.model_validate(result.data)
  if len(bundle.evidences) < 3:
    raise ValueError("Insufficient evidence; refine query or widen scope")
  return bundle
```

### Example — Synthesis with Structured Output
```python
class Citation(BaseModel):
  url: HttpUrl
  label: str

class Section(BaseModel):
  title: str
  content_md: str
  citations: list[Citation]

class MRDOutput(BaseModel):
  topic: str
  sections: list[Section]
  summary: str

writer = Agent(
  model="gemini-1.5-pro",
  system_prompt=(
    "Write an MRD with sections and citations. Output JSON matching MRDOutput."
  ),
)

def synthesize(bundle: ResearchBundle) -> MRDOutput:
  result = writer.run(
    {"topic": bundle.query, "evidence": [e.model_dump() for e in bundle.evidences]},
    options=RunOptions(response_model=MRDOutput)
  )
  mrd = MRDOutput.model_validate(result.data)
  # Structural checks can be applied here as well.
  return mrd
```

## Return Formats & Contracts
- **Markdown**: Human-readable MRD stored under [output/](output/), following naming convention with IDs and timestamps.
- **JSON**: Machine-readable artifact mirroring `MRDOutput` for downstream automations.
- **Citations**: Each claim is traceable to `Evidence` via URL and label; broken links or missing citations fail validation.

## Operational Flow — Orchestrator Hooks
- **INIT → PLANNING**: Validate `PromptSpec`; normalize and derive goals.
- **PLANNING → RESEARCH**: Select agents; call adapters; collect `ResearchBundle`.
- **RESEARCH → SYNTHESIS**: Pass bundle to writer; enforce `MRDOutput` schema.
- **VALIDATION → FORMAT**: Run safety, structure, and factuality gates; format Markdown/JSON.
- **DELIVERY → DONE**: Persist artifacts; emit logs/metrics/traces; return response.

