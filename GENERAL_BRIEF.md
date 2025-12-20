# Autonomous MRD Agent – Interview Brief

## What this agent does
- Takes a high-level intent and produces a validated, database-ready Market Requirements Document (MRD) for the real-money skill gaming vertical (Triumph case), using the architecture implemented in [main.py](main.py), [orchestration/state_machine.py](orchestration/state_machine.py), [models/core.py](models/core.py), and [models/mrd.py](models/mrd.py).
- Guarantees structure and traceability by pushing all intermediate and final data through Pydantic models with validators (e.g., every `VerifiedClaim` requires citations; HIGH confidence requires multiple citations).
- Uses a finite state machine (FSM) to govern progress from Planning → Research → Synthesis → Validation → (optional Human Review) → Complete. The FSM enforces data gates (completeness thresholds) to avoid hallucinated synthesis.

## End-to-end flow (who does what)
- **Planning** ([planning_handler](main.py#L29)): Interprets the user prompt and emits a `ResearchPlan` of typed `ResearchTask`s (task_id, task_type, required_tools, success_criteria). Natural HITL checkpoint to approve/edit tasks.
	- Key helpers: `ResearchTask`/`ResearchPlan` validators in [models/core.py](models/core.py) block duplicate task_ids and enforce required fields.

- **Research** ([research_handler](main.py#L93)): Iterates `research_plan.tasks` and dispatches by `task_type`.
	- Competitor: [CompetitorAnalysisAgent.run](agents/research_agents.py#L194) → [analyze_competitor](agents/research_agents.py#L163) → `MockToolkit.search_sensor_tower` through [execute_with_retry_and_fallback](agents/research_agents.py#L36); emits `CompetitorProfile` plus cited `VerifiedClaim`.
	- Sentiment: [SentimentAnalysisAgent.analyze_platform](agents/research_agents.py#L207) → `MockToolkit.analyze_sentiment` with retries/fallbacks; emits `SentimentAnalysis`.
	- Regulatory: [RegulatoryAnalysisAgent.check_region](agents/research_agents.py#L243) → `MockToolkit.check_regulatory_compliance`; maps payload to `RegulatoryStatus`, downgrades to UNVERIFIED on failures.
	- Gap: inline mock in handler populates `GapAnalysisItem` with cited `VerifiedClaim`.
	- Aggregation: results land in `ResearchAggregate`; failures go to `failed_tasks`. [ResearchAggregate.calculate_completeness](models/core.py#L219) recomputes coverage for the synthesis gate.

- **Synthesis** ([synthesis_handler](main.py#L163)): Runs only if [can_transition_to_synthesis](orchestration/state_machine.py#L90) passes (completeness threshold + critical checks like competitor presence). Builds `StrategicAnalysis` MRD, composing:
	- SWOT: `SWOTAnalysis` built from `VerifiedClaim`s with citations.
	- Competitive/GTM/Features/Risks: Uses `ResearchAggregate` data to populate MRD fields, all Pydantic-enforced.

- **Validation** ([validation_handler](main.py#L303)): Audits the MRD:
	- `check_claims` scans `VerifiedClaim`s counting LOW/UNVERIFIED.
	- Ensures critical sections (competitor_list, regulatory_analysis) exist.
	- Emits `MRDValidationResult` (approve|revise|reject) which the FSM reads via [can_transition_to_complete](orchestration/state_machine.py#L121).

- **Human Review**: Optional stop where a reviewer can approve/override after validation (maps to `AgentState.HUMAN_REVIEW`).

## Why a finite state machine (FSM)?
- **Deterministic control**: Explicit transitions and guards replace brittle chain-of-thought prompting; every step has entry/exit conditions.
- **Hallucination brakes**: `can_transition_to_synthesis` checks data completeness and presence of critical sections (e.g., competitor data) before allowing generation.
- **Auditability**: `StateTransition` log records from→to, timestamp, reason—clear trace for debugging or compliance.
- **Interruptibility**: Natural pause points for HITL approval (research plan, MRD validation).

## How Pydantic enforces contract (and prevents “vibe” output)
- **Input contracts**: `ResearchTask`/`ResearchPlan` in [models/core.py](models/core.py) define what research must produce (task_type, required_tools, success_criteria, retries/timeouts). Duplicate task IDs are rejected by a validator.
- **Evidence discipline**: `VerifiedClaim` requires citations; HIGH confidence demands ≥2 citations (validator). Confidence is explicit (`ConfidenceLevel`).
- **Output contract**: `StrategicAnalysis` in [models/mrd.py](models/mrd.py) is the MRD schema. Missing required fields or mismatched types fail fast before any DB write.
- **Quality gates**: `MRDValidationResult` captures errors/warnings and drives FSM decisions.

## Tooling layer and failure handling
- **Mock interfaces** in [agents/research_agents.py](agents/research_agents.py): `search_sensor_tower`, `analyze_sentiment`, `check_regulatory_compliance`, `web_search` all return a normalized `ToolResponse` with `success`, `data`, `error_message`, `raw_response` hashable for audit.
- **Retries**: `execute_with_retry` backs off on failures, returning a structured failure if all attempts exhaust.
- **Fallbacks & visibility**: Failed tasks are appended to `failed_tasks`; completeness score drops, blocking synthesis until coverage is adequate. Errors accumulate in context for observability.
- **Example: Sensor Tower returns no data**: Competitor agent returns `None`; task is marked failed; completeness < threshold ⇒ FSM stays in Research. No synthetic competitor claims are generated, avoiding hallucination.

## Anti-hallucination controls
- Structured schemas at every hop (Pydantic) prevent free-form text from slipping through.
- Citations required on claims; HIGH confidence requires multiple sources.
- Data completeness gate before synthesis; validation gate before completion.
- Human-in-loop checkpoints for plan approval and MRD review.

## Function-by-function highlights (practical talking points)
- **`Orchestrator.run`** (state loop) in [orchestration/state_machine.py](orchestration/state_machine.py): Drives iteration until COMPLETE/ERROR; applies transition rules after each handler.
- **`Orchestrator.run`** (state loop) in [orchestration/state_machine.py](orchestration/state_machine.py#L202): Drives iteration until COMPLETE/ERROR; applies transition rules after each handler.
- **`StateTransitionRules.can_transition_to_research/synthesis/complete`**: Readiness gates at [L76](orchestration/state_machine.py#L76), [L90](orchestration/state_machine.py#L90), [L121](orchestration/state_machine.py#L121).
- **`CompetitorAnalysisAgent.analyze_competitor`**: [agents/research_agents.py#L163](agents/research_agents.py#L163) calls Sensor Tower via fallback, emits `CompetitorProfile` with citations.
- **`SentimentAnalysisAgent.analyze_platform`**: [agents/research_agents.py#L207](agents/research_agents.py#L207) wraps sentiment tool with retry/fallback; returns ratios + influencers.
- **`RegulatoryAnalysisAgent.check_region`**: [agents/research_agents.py#L243](agents/research_agents.py#L243) wraps regulatory tool; downgrades to UNVERIFIED on failure.
- **`ResearchAggregate.calculate_completeness`**: [models/core.py#L219](models/core.py#L219) scores coverage across dimensions for the synthesis gate.
- **`synthesis_handler`**: [main.py#L163](main.py#L163) assembles SWOT, GTM, features, risks into `StrategicAnalysis` with citations.
- **`validation_handler`**: [main.py#L303](main.py#L303) counts LOW/UNVERIFIED claims, ensures critical sections exist, sets recommendation (`approve|revise|reject`).

## How to modularize for SaaS (swap verticals quickly)
- **Keep the orchestration & schemas**: FSM, transition rules, and MRD structure stay the same.
- **Swap task definitions**: In Planning, emit SaaS-specific `ResearchTask`s (e.g., pricing analysis, ICP validation, churn benchmarks). The same `ResearchPlan`/`ResearchAggregate` scaffolding applies.
- **Extend agents**: Add domain-specific agents (e.g., `PricingAnalysisAgent`, `ChurnBenchmarksAgent`) that still return typed objects and citations. Reuse `execute_with_retry` and `ToolResponse` contracts.
- **Reuse validation gates**: Update `can_transition_to_synthesis` thresholds to reflect SaaS-critical signals (e.g., competitor pricing, ARR benchmarks). Keep hallucination controls identical.
- **Benefit**: Only the task catalog and tool adapters change; the FSM, safety rails, and MRD schema continue to enforce structure and data quality.



## One-line rationale
Structured FSM + Pydantic-first contracts = deterministic, auditable, and citation-backed MRDs that can swap verticals with minimal change.
