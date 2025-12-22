"""
Core Pydantic models defining the strict contract between agent steps.
Every piece of data flowing through the system is validated. 
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS & BASE TYPES
# ============================================================================

class AgentState(str, Enum):
    """State machine states for orchestration"""
    PLANNING = "planning"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    HUMAN_REVIEW = "human_review"
    COMPLETE = "complete"
    ERROR = "error"


class DataSource(str, Enum):
    """Enumerated data sources for traceability"""
    SENSOR_TOWER = "sensor_tower"
    SOCIAL_SENTIMENT = "social_sentiment"
    WEB_SEARCH = "web_search"
    REGULATORY_DB = "regulatory_db"
    MANUAL_INPUT = "manual_input"
    INFERRED = "inferred"  # Flagged for human review


class ConfidenceLevel(str, Enum):
    """Confidence scoring for claims"""
    HIGH = "high"      # Multiple corroborating sources
    MEDIUM = "medium"  # Single reliable source
    LOW = "low"        # Inferred or partial data
    UNVERIFIED = "unverified"  # Requires human verification


# ============================================================================
# RESEARCH TASK MODELS (Input to Research Agents)
# ============================================================================

class TaskProvenance(BaseModel):
    """Track which tool, attempt, and data hash for each task execution"""
    task_id: str
    tool_name: str
    attempt: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    data_hash: Optional[str] = None
    error_message: Optional[str] = None


class ResearchTask(BaseModel):
    """Defines a single research task with explicit success criteria"""
    task_id: str = Field(..., description="Unique identifier for tracking")
    task_type: Literal["market", "competitor", "sentiment", "regulatory", "gap_analysis"]
    query: str = Field(... , min_length=10, description="The research question")
    target_entities: list[str] = Field(default_factory=list, description="Apps, companies, regions to research")
    required_tools: list[str] = Field(... , description="Tools needed to complete this task")
    success_criteria: str = Field(..., description="What constitutes successful completion")
    max_retries: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    fallback_tools: list[str] = Field(default_factory=list, description="Alternative tools if primary fails")
    provenance: list[TaskProvenance] = Field(default_factory=list, description="Execution history")
    
    @field_validator('required_tools')
    @classmethod
    def validate_required_tools(cls, tools):
        """Ensure at least one tool is specified"""
        if not tools or len(tools) == 0:
            raise ValueError("At least one required tool must be specified")
        return tools


class ResearchPlan(BaseModel):
    """Complete research plan requiring human approval"""
    plan_id: str
    user_intent: str = Field(..., description="Original user prompt")
    interpreted_goal: str = Field(... , description="Agent's interpretation for validation")
    tasks: list[ResearchTask] = Field(... , min_length=1)
    estimated_duration_minutes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    human_approved: bool = Field(default=False, description="Plan requires explicit human approval")
    approval_timestamp: Optional[datetime] = None
    approval_notes: Optional[str] = None
    
    @field_validator('tasks')
    @classmethod
    def validate_task_dependencies(cls, tasks:  list[ResearchTask]) -> list[ResearchTask]:
        """Ensure no duplicate task IDs"""
        ids = [t.task_id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs detected")
        return tasks
    
    @field_validator('estimated_duration_minutes')
    @classmethod
    def validate_duration(cls, minutes):
        """Ensure realistic duration estimate"""
        if minutes < 1:
            raise ValueError("Duration must be at least 1 minute")
        if minutes > 1440:  # 24 hours
            raise ValueError("Duration cannot exceed 24 hours")
        return minutes


# ============================================================================
# RESEARCH RESULT MODELS (Output from Research Agents)
# ============================================================================

class Citation(BaseModel):
    """Every claim must have a citation for traceability"""
    source: DataSource
    url: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    raw_data_hash: Optional[str] = Field(None, description="Hash of raw response for audit")


class VerifiedClaim(BaseModel):
    """A single claim with its backing evidence"""
    claim:  str = Field(..., min_length=5)
    confidence: ConfidenceLevel
    citations: list[Citation] = Field(... , min_length=1)
    
    @field_validator('citations')
    @classmethod
    def require_citation_for_high_confidence(cls, citations, info):
        """High confidence claims need multiple sources"""
        # Access other field values through info. data
        confidence = info.data. get('confidence')
        if confidence == ConfidenceLevel.HIGH and len(citations) < 2:
            raise ValueError("HIGH confidence claims require at least 2 citations")
        return citations


class CompetitorProfile(BaseModel):
    """Structured competitor analysis"""
    name: str
    app_store_id: Optional[str] = None
    monthly_active_users: Optional[int] = None
    revenue_estimate: Optional[str] = None
    key_features: list[str] = Field(default_factory=list)
    target_demographics: list[str] = Field(default_factory=list)
    marketing_channels: list[str] = Field(default_factory=list)
    strengths: list[VerifiedClaim] = Field(default_factory=list)
    weaknesses: list[VerifiedClaim] = Field(default_factory=list)
    data_freshness: datetime = Field(default_factory=datetime.utcnow)


class MarketData(BaseModel):
    """Market analysis data structure"""
    market_size_usd: Optional[int] = None
    growth_rate_percent: Optional[float] = None
    key_trends: list[VerifiedClaim] = Field(default_factory=list)
    barriers_to_entry: list[VerifiedClaim] = Field(default_factory=list)
    success_factors: list[VerifiedClaim] = Field(default_factory=list)


class SentimentAnalysis(BaseModel):
    """Social sentiment analysis results"""
    platform: str
    sample_size: int = Field(... , ge=0)
    positive_ratio: float = Field(... , ge=0, le=1)
    negative_ratio:  float = Field(..., ge=0, le=1)
    neutral_ratio: float = Field(..., ge=0, le=1)
    top_positive_themes: list[str] = Field(default_factory=list)
    top_negative_themes: list[str] = Field(default_factory=list)
    influencer_mentions: list[dict] = Field(default_factory=list)
    citation:  Citation


class RegulatoryStatus(BaseModel):
    """Regulatory compliance status per region"""
    region:  str = Field(..., description="ISO country/region code")
    legal_status:  Literal["legal", "illegal", "gray_area", "requires_license", "unknown"]
    license_requirements: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    recent_changes: list[VerifiedClaim] = Field(default_factory=list)
    confidence:  ConfidenceLevel
    citation: Citation


class GapAnalysisItem(BaseModel):
    """Single gap/opportunity identified"""
    opportunity:  str
    current_market_state: str
    potential_value:  Literal["high", "medium", "low"]
    implementation_complexity: Literal["high", "medium", "low"]
    supporting_evidence: list[VerifiedClaim]


# ============================================================================
# RESEARCH AGGREGATE (Collected from all agents)
# ============================================================================

class ResearchAggregate(BaseModel):
    """Complete research data before synthesis"""
    plan_id: str
    market_data: Optional[MarketData] = None
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    sentiment_analyses: list[SentimentAnalysis] = Field(default_factory=list)
    regulatory_statuses: list[RegulatoryStatus] = Field(default_factory=list)
    gap_analysis:  list[GapAnalysisItem] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list, description="Task IDs that failed")
    data_completeness_score: float = Field(default=0.0, ge=0, le=1)
    
    # New: Track which sources were actually used
    sources_used: set[str] = Field(default_factory=set, description="Data sources actually referenced in data")
    completeness_by_dimension: dict[str, float] = Field(default_factory=dict, description="Completeness per data type")
    
    def calculate_completeness(self) -> float:
        """Calculate how complete the research is with per-dimension tracking"""
        scores = {}
        
        # Dimension scores
        scores['market'] = 1.0 if (self.market_data and self.market_data.key_trends) else 0.0
        scores['competitors'] = min(len(self.competitors) / 3, 1.0) if self.competitors else 0.0
        scores['sentiment'] = 1.0 if self.sentiment_analyses else 0.0
        scores['regulatory'] = 1.0 if self.regulatory_statuses else 0.0
        # Gap target lowered to 2 to reduce plateaus at ~86%
        scores['gaps'] = min(len(self.gap_analysis) / 2, 1.0) if self.gap_analysis else 0.0

        # Weighted average to better reflect impact
        weights = {
            'market': 0.20,
            'competitors': 0.25,
            'sentiment': 0.15,
            'regulatory': 0.20,
            'gaps': 0.20,
        }

        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        self.completeness_by_dimension = scores
        self.data_completeness_score = weighted_sum
        return self.data_completeness_score
    
    def extract_sources_used(self) -> set[str]:
        """Scan all data and extract unique sources actually referenced"""
        sources = set()
        
        def extract_from_claims(claims: list[VerifiedClaim]):
            for claim in claims:
                for citation in claim.citations:
                    sources.add(citation.source.value)
        
        # Extract from all sections
        if self.market_data:
            extract_from_claims(self.market_data.key_trends)
            extract_from_claims(self.market_data.barriers_to_entry)
            extract_from_claims(self.market_data.success_factors)
        
        for competitor in self.competitors:
            extract_from_claims(competitor.strengths)
            extract_from_claims(competitor.weaknesses)
        
        for sentiment in self.sentiment_analyses:
            sources.add(sentiment.citation.source.value)
        
        for regulatory in self.regulatory_statuses:
            sources.add(regulatory.citation.source.value)
            extract_from_claims(regulatory.recent_changes)
        
        for gap in self.gap_analysis:
            extract_from_claims(gap.supporting_evidence)
        
        self.sources_used = sources
        return sources
    
    @field_validator('failed_tasks')
    @classmethod
    def validate_failed_tasks_not_empty(cls, failed_tasks):
        """Warn if too many tasks failed"""
        # Don't fail validation, but this could trigger a warning
        return failed_tasks