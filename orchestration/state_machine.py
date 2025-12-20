"""
State Machine Orchestrator - The "Brain" of the agent. 
Manages transitions between states and determines when to proceed. 
"""

from __future__ import annotations
import asyncio
import uuid
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from models.core import (
    AgentState, ResearchPlan, ResearchTask, ResearchAggregate,
    ConfidenceLevel
)
from models.mrd import StrategicAnalysis, MRDValidationResult


class StateTransition(BaseModel):
    """Record of a state transition for debugging/audit"""
    from_state: AgentState
    to_state: AgentState
    timestamp: datetime
    reason: str
    metadata: dict = {}


class OrchestratorContext(BaseModel):
    """
    Central context object passed through the pipeline.
    This is how the agent "remembers" what it's doing.
    """
    session_id: str
    current_state: AgentState = AgentState. PLANNING
    user_prompt: str
    research_plan: Optional[ResearchPlan] = None
    research_aggregate: Optional[ResearchAggregate] = None
    mrd:  Optional[StrategicAnalysis] = None
    validation_result: Optional[MRDValidationResult] = None
    
    # Tracking
    transitions: list[StateTransition] = []
    errors: list[str] = []
    retry_counts: dict[str, int] = {}
    
    # Human-in-the-loop (HITL) Approval Flags
    plan_approved_by_human: bool = False
    plan_approval_timestamp: Optional[datetime] = None
    plan_reviewer_notes: Optional[str] = None
    
    synthesis_approved_by_human: bool = False
    synthesis_approval_timestamp: Optional[datetime] = None
    synthesis_reviewer_notes: Optional[str] = None
    
    # Thresholds
    min_completeness_for_synthesis: float = 0.6
    max_retries_per_task: int = 3
    
    # Lifecycle tracking
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    last_state_entry: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True


class StateTransitionRules: 
    """
    Defines the rules for transitioning between states. 
    This is where the "intelligence" of knowing when to proceed lives.
    """
    
    @staticmethod
    def can_transition_to_research(ctx: OrchestratorContext) -> tuple[bool, str]: 
        """Check if we can move from PLANNING to RESEARCH"""
        if ctx. research_plan is None: 
            return False, "No research plan generated"
        if len(ctx.research_plan.tasks) == 0:
            return False, "Research plan has no tasks"
        
        # HITL check: plan must be approved
        if not ctx.plan_approved_by_human:
            return False, "Research plan requires human approval before proceeding"
        
        return True, "Research plan approved by human"
    
    @staticmethod
    def can_transition_to_synthesis(ctx:  OrchestratorContext) -> tuple[bool, str]: 
        """
        Check if we have enough data to synthesize. 
        This is KEY to avoiding "hallucination" - we don't proceed without data.
        """
        if ctx.research_aggregate is None: 
            return False, "No research data collected"
        
        completeness = ctx.research_aggregate.calculate_completeness()
        
        if completeness < ctx. min_completeness_for_synthesis: 
            return False, f"Data completeness {completeness:.1%} below threshold {ctx.min_completeness_for_synthesis:. 1%}"
        
        # Check for critical missing data - these are hard blockers
        if not ctx.research_aggregate.competitors:
            return False, "No competitor data - critical for analysis"
        
        if not ctx.research_aggregate.regulatory_statuses:
            return False, "No regulatory data - critical for compliance analysis"
        
        if not ctx.research_aggregate.sentiment_analyses:
            return False, "No sentiment data - critical for audience analysis"
        
        # Check completeness by dimension
        dim_completeness = ctx.research_aggregate.completeness_by_dimension
        if dim_completeness.get('competitors', 0) < 0.33:
            return False, f"Insufficient competitor data: {dim_completeness.get('competitors', 0):.1%}"
        
        return True, f"Data completeness {completeness:.1%} - sufficient for synthesis"
    
    @staticmethod
    def can_transition_to_complete(ctx: OrchestratorContext) -> tuple[bool, str]:
        """Check if MRD passes validation"""
        if ctx.validation_result is None: 
            return False, "MRD not validated"
        if not ctx.validation_result.is_valid:
            return False, f"MRD validation failed: {'; '.join(ctx.validation_result.errors)}"
        if ctx.validation_result.recommendation == "reject":
            return False, "MRD rejected by validator"
        return True, "MRD validated and approved"


class Orchestrator:
    """
    Main orchestration engine implementing a state machine pattern.
    """
    
    def __init__(self):
        self.handlers: dict[AgentState, Callable[[OrchestratorContext], Awaitable[OrchestratorContext]]] = {}
        self.rules = StateTransitionRules()
    
    def register_handler(
        self, 
        state:  AgentState, 
        handler:  Callable[[OrchestratorContext], Awaitable[OrchestratorContext]]
    ):
        """Register a handler for a specific state"""
        self. handlers[state] = handler
    
    def _transition(self, ctx: OrchestratorContext, new_state: AgentState, reason: str):
        """Execute a state transition with logging"""
        transition = StateTransition(
            from_state=ctx.current_state,
            to_state=new_state,
            timestamp=datetime.utcnow(),
            reason=reason
        )
        ctx.transitions.append(transition)
        ctx.current_state = new_state
        print(f"[TRANSITION] {transition.from_state} -> {transition.to_state}:  {reason}")
    
    def _determine_next_state(self, ctx: OrchestratorContext) -> tuple[AgentState, str]:
        """
        Determine the next state based on current context.
        This is the "decision engine" of the state machine.
        """
        current = ctx.current_state
        
        if current == AgentState.PLANNING: 
            can_proceed, reason = self. rules.can_transition_to_research(ctx)
            if can_proceed: 
                return AgentState.RESEARCH, reason
            return AgentState. PLANNING, reason  # Stay in planning
        
        elif current == AgentState.RESEARCH: 
            can_proceed, reason = self.rules.can_transition_to_synthesis(ctx)
            if can_proceed:
                return AgentState. SYNTHESIS, reason
            # Check if we've exhausted retries
            if len(ctx.errors) > 10:
                return AgentState.ERROR, "Too many errors during research"
            return AgentState.RESEARCH, reason  # Continue research
        
        elif current == AgentState.SYNTHESIS: 
            if ctx.mrd is not None:
                return AgentState. VALIDATION, "MRD generated"
            return AgentState.SYNTHESIS, "Still synthesizing"
        
        elif current == AgentState.VALIDATION:
            can_proceed, reason = self.rules.can_transition_to_complete(ctx)
            if can_proceed:
                return AgentState.COMPLETE, reason
            if ctx.validation_result and ctx.validation_result.recommendation == "revise":
                return AgentState. SYNTHESIS, "Revisions needed"
            return AgentState. HUMAN_REVIEW, "Requires human review"
        
        elif current == AgentState. HUMAN_REVIEW: 
            # In production: check for human input
            return AgentState.COMPLETE, "Human approved"
        
        return AgentState.ERROR, "Unknown state"
    
    async def run(self, user_prompt: str) -> OrchestratorContext:
        """
        Main execution loop. 
        Continues until COMPLETE or ERROR state is reached.
        """
        ctx = OrchestratorContext(
            session_id=str(uuid.uuid4()),
            user_prompt=user_prompt
        )
        
        max_iterations = 50  # Safety limit
        iteration = 0
        
        while ctx.current_state not in [AgentState.COMPLETE, AgentState.ERROR]: 
            iteration += 1
            if iteration > max_iterations: 
                self._transition(ctx, AgentState.ERROR, "Max iterations exceeded")
                break
            
            # Execute handler for current state
            handler = self.handlers.get(ctx. current_state)
            if handler: 
                try:
                    ctx = await handler(ctx)
                except Exception as e:
                    ctx.errors. append(f"Handler error in {ctx.current_state}:  {str(e)}")
                    print(f"[ERROR] {e}")
            
            # Determine next state
            next_state, reason = self._determine_next_state(ctx)
            
            if next_state != ctx.current_state:
                self._transition(ctx, next_state, reason)
        
        return ctx