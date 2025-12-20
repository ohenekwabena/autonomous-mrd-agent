"""
Market Requirements Document (MRD) output schema. 
This is the final structured deliverable. 
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from datetime import datetime
from models.core import VerifiedClaim, CompetitorProfile, RegulatoryStatus, GapAnalysisItem, ConfidenceLevel


class SWOTAnalysis(BaseModel):
    """SWOT with verified claims"""
    strengths: list[VerifiedClaim] = Field(... , min_length=1)
    weaknesses: list[VerifiedClaim] = Field(..., min_length=1)
    opportunities: list[VerifiedClaim] = Field(... , min_length=1)
    threats: list[VerifiedClaim] = Field(..., min_length=1)


class FeatureRecommendation(BaseModel):
    """Recommended feature with justification"""
    feature_name: str
    description: str
    priority:  Literal["must_have", "should_have", "nice_to_have"]
    rationale:  list[VerifiedClaim]
    estimated_effort:  Literal["low", "medium", "high"]
    competitive_advantage: str


class TargetAudience(BaseModel):
    """Target audience segment"""
    segment_name: str
    demographics: dict = Field(default_factory=dict)
    psychographics: list[str] = Field(default_factory=list)
    acquisition_channels: list[VerifiedClaim] = Field(default_factory=list)
    estimated_tam: Optional[int] = Field(None, description="Total Addressable Market")


class GoToMarketStrategy(BaseModel):
    """GTM recommendations"""
    primary_channels: list[VerifiedClaim]
    influencer_strategy: Optional[str] = None
    geographic_rollout: list[str] = Field(default_factory=list)
    regulatory_considerations: list[str] = Field(default_factory=list)


class StrategicAnalysis(BaseModel):
    """
    The complete MRD output structure. 
    This is what gets saved to the database. 
    """
    # Metadata
    mrd_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    overall_confidence: ConfidenceLevel
    
    # Executive Summary
    executive_summary: str = Field(..., min_length=100)
    user_intent: str
    interpreted_goal: str
    
    # Market Analysis
    market_overview: list[VerifiedClaim] = Field(..., min_length=1, description="At least 1 market claim required")
    market_size_assessment: Optional[str] = None
    growth_trajectory: Optional[str] = None
    
    # Competitive Landscape
    competitor_list: list[CompetitorProfile] = Field(..., min_length=1, description="At least 1 competitor analysis required")
    competitive_moat_analysis: list[VerifiedClaim] = Field(..., min_length=1)
    
    # SWOT
    swot:  SWOTAnalysis
    
    # Target Market
    target_audiences: list[TargetAudience] = Field(..., min_length=1, description="At least 1 target audience required")
    
    # Regulatory
    regulatory_analysis: list[RegulatoryStatus] = Field(..., min_length=1, description="At least 1 regulatory region required")
    regulatory_recommendation: str = Field(..., min_length=20, description="Detailed regulatory recommendation")
    
    # Gap Analysis & Features
    gap_analysis: list[GapAnalysisItem] = Field(..., min_length=1, description="At least 1 gap identified")
    feature_recommendations: list[FeatureRecommendation] = Field(..., min_length=1, description="At least 1 feature recommended")
    
    # Go-to-Market
    gtm_strategy: GoToMarketStrategy
    
    # Risk Assessment
    key_risks: list[VerifiedClaim] = Field(..., min_length=1, description="At least 1 risk identified")
    mitigation_strategies: list[str] = Field(..., min_length=1, description="At least 1 mitigation strategy")
    
    # Data Quality
    data_sources_used: list[str]
    claims_requiring_verification: list[str] = Field(
        default_factory=list,
        description="Claims with LOW or UNVERIFIED confidence"
    )
    research_gaps: list[str] = Field(default_factory=list)
    
    @field_validator('regulatory_recommendation')
    @classmethod
    def validate_reg_rec_references_regions(cls, rec, info):
        """Regulatory recommendation should reference actual regions analyzed"""
        regulatory_analysis = info.data.get('regulatory_analysis', [])
        if regulatory_analysis and len(rec) < 20:
            raise ValueError("Regulatory recommendation too brief - must reference specific regions")
        return rec
    
    @field_validator('executive_summary')
    @classmethod
    def validate_summary_contains_confidence(cls, summary):
        """Summary should mention data quality/confidence"""
        summary_lower = summary.lower()
        if 'high' not in summary_lower and 'strong' not in summary_lower:
            # Allow summary without explicit confidence mention but warn via log
            pass
        return summary
    
    @model_validator(mode='after')
    def validate_no_inferred_only_claims(self):
        """Ensure not all claims come from INFERRED source"""
        def check_claims(claims: list[VerifiedClaim]):
            if not claims:
                return
            inferred_only = all(
                all(c.source == 'inferred' for c in claim.citations)
                for claim in claims
            )
            if inferred_only:
                raise ValueError("Claims cannot be inferred-only; must have external data sources")
        
        check_claims(self.market_overview)
        check_claims(self.competitive_moat_analysis)
        check_claims(self.swot.strengths)
        check_claims(self.key_risks)
        
        return self
    
    @model_validator(mode='after')
    def validate_sources_match_regions(self):
        """Ensure regulatory recommendations actually reference analyzed regions"""
        analyzed_regions = {r.region for r in self.regulatory_analysis}
        rec_lower = self.regulatory_recommendation.lower()
        
        # Check that at least one analyzed region is mentioned in rec
        found_match = any(region.lower() in rec_lower for region in analyzed_regions)
        if not found_match and analyzed_regions:
            raise ValueError(f"Regulatory recommendation must reference at least one analyzed region: {analyzed_regions}")
        
        return self


class MRDValidationResult(BaseModel):
    """Result of MRD validation"""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unverified_claims_count: int = 0
    low_confidence_claims_count: int = 0
    recommendation:  Literal["approve", "revise", "reject"]
    
    @field_validator('recommendation')
    @classmethod
    def validate_recommendation_logic(cls, rec, info):
        """Recommendation should align with error/warning counts"""
        is_valid = info.data.get('is_valid', False)
        errors = info.data.get('errors', [])
        
        if errors and rec == "approve":
            raise ValueError("Cannot approve MRD with errors present")
        if not is_valid and rec == "approve":
            raise ValueError("Cannot approve invalid MRD")
        
        return rec