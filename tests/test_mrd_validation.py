"""
Test suite for MRD validation.
Ensures strict validation of the final output document.
"""

import pytest
from datetime import datetime
import json

from models.core import (
    VerifiedClaim, Citation, DataSource, ConfidenceLevel,
    CompetitorProfile, RegulatoryStatus
)
from models.mrd import (
    StrategicAnalysis, SWOTAnalysis, FeatureRecommendation,
    TargetAudience, GoToMarketStrategy, MRDValidationResult
)


class TestMRDStructuralValidation:
    """Test that MRD enforces required structure"""
    
    def test_mrd_requires_market_overview(self):
        """MRD must have market overview claims"""
        with pytest.raises(ValueError, match="min_length"):
            StrategicAnalysis(
                mrd_id="mrd_001",
                overall_confidence=ConfidenceLevel.HIGH,
                executive_summary="This is a long executive summary that meets minimum length requirement of 100 characters for validation",
                user_intent="Build an app",
                interpreted_goal="Analyze market",
                market_overview=[],  # Empty - invalid!
                competitor_list=[CompetitorProfile(name="Triumph")],
                competitive_moat_analysis=[
                    VerifiedClaim(
                        claim="Strong retention",
                        confidence=ConfidenceLevel.HIGH,
                        citations=[
                            Citation(source=DataSource.SENSOR_TOWER),
                            Citation(source=DataSource.WEB_SEARCH)
                        ]
                    )
                ],
                swot=SWOTAnalysis(
                    strengths=[VerifiedClaim(
                        claim="Strong market position",
                        confidence=ConfidenceLevel.HIGH,
                        citations=[
                            Citation(source=DataSource.SENSOR_TOWER),
                            Citation(source=DataSource.WEB_SEARCH)
                        ]
                    )],
                    weaknesses=[VerifiedClaim(
                        claim="High regulatory risk",
                        confidence=ConfidenceLevel.MEDIUM,
                        citations=[Citation(source=DataSource.REGULATORY_DB)]
                    )],
                    opportunities=[VerifiedClaim(
                        claim="Emerging IO game segment",
                        confidence=ConfidenceLevel.MEDIUM,
                        citations=[Citation(source=DataSource.WEB_SEARCH)]
                    )],
                    threats=[VerifiedClaim(
                        claim="Regulatory tightening",
                        confidence=ConfidenceLevel.MEDIUM,
                        citations=[Citation(source=DataSource.REGULATORY_DB)]
                    )]
                ),
                target_audiences=[TargetAudience(segment_name="Young Males")],
                regulatory_analysis=[
                    RegulatoryStatus(
                        region="UK",
                        legal_status="requires_license",
                        confidence=ConfidenceLevel.HIGH,
                        citation=Citation(source=DataSource.REGULATORY_DB)
                    )
                ],
                regulatory_recommendation="License in UK, consider Malta",
                gap_analysis=[],
                feature_recommendations=[
                    FeatureRecommendation(
                        feature_name="IO Game Mode",
                        description="Agar.io-style games",
                        priority="should_have",
                        rationale=[VerifiedClaim(
                            claim="Market opportunity",
                            confidence=ConfidenceLevel.MEDIUM,
                            citations=[Citation(source=DataSource.WEB_SEARCH)]
                        )],
                        estimated_effort="medium",
                        competitive_advantage="First-mover"
                    )
                ],
                gtm_strategy=GoToMarketStrategy(
                    primary_channels=[VerifiedClaim(
                        claim="TikTok",
                        confidence=ConfidenceLevel.HIGH,
                        citations=[
                            Citation(source=DataSource.SOCIAL_SENTIMENT),
                            Citation(source=DataSource.WEB_SEARCH)
                        ]
                    )]
                ),
                key_risks=[VerifiedClaim(
                    claim="Regulatory change",
                    confidence=ConfidenceLevel.MEDIUM,
                    citations=[Citation(source=DataSource.REGULATORY_DB)]
                )],
                mitigation_strategies=["Diversify channels"],
                data_sources_used=["sensor_tower", "regulatory_db"]
            )
    
    def test_mrd_requires_competitors(self):
        """MRD must have at least 1 competitor analysis"""
        with pytest.raises(ValueError, match="min_length"):
            StrategicAnalysis(
                mrd_id="mrd_001",
                overall_confidence=ConfidenceLevel.HIGH,
                executive_summary="This is a long executive summary that meets minimum length requirement",
                user_intent="Build an app",
                interpreted_goal="Analyze market",
                market_overview=[VerifiedClaim(
                    claim="Growing market",
                    confidence=ConfidenceLevel.MEDIUM,
                    citations=[Citation(source=DataSource.WEB_SEARCH)]
                )],
                competitor_list=[],  # Empty - invalid!
                competitive_moat_analysis=[VerifiedClaim(
                    claim="Strong moat",
                    confidence=ConfidenceLevel.MEDIUM,
                    citations=[Citation(source=DataSource.SENSOR_TOWER)]
                )],
                swot=SWOTAnalysis(
                    strengths=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    weaknesses=[VerifiedClaim(claim="y", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    opportunities=[VerifiedClaim(claim="z", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    threats=[VerifiedClaim(claim="a", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
                ),
                target_audiences=[TargetAudience(segment_name="Gamers")],
                regulatory_analysis=[RegulatoryStatus(
                    region="UK",
                    legal_status="legal",
                    confidence=ConfidenceLevel.MEDIUM,
                    citation=Citation(source=DataSource.REGULATORY_DB)
                )],
                regulatory_recommendation="OK to launch",
                gap_analysis=[],
                feature_recommendations=[FeatureRecommendation(
                    feature_name="Feature",
                    description="Desc",
                    priority="nice_to_have",
                    rationale=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    estimated_effort="low",
                    competitive_advantage="x"
                )],
                gtm_strategy=GoToMarketStrategy(
                    primary_channels=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
                ),
                key_risks=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                mitigation_strategies=["x"],
                data_sources_used=[]
            )
    
    def test_mrd_requires_regulatory_analysis(self):
        """MRD must have regulatory analysis for at least 1 region"""
        with pytest.raises(ValueError, match="min_length"):
            StrategicAnalysis(
                mrd_id="mrd_001",
                overall_confidence=ConfidenceLevel.HIGH,
                executive_summary="This is a long executive summary that meets minimum length requirement",
                user_intent="Build an app",
                interpreted_goal="Analyze market",
                market_overview=[VerifiedClaim(claim="Growing", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                competitor_list=[CompetitorProfile(name="Competitor")],
                competitive_moat_analysis=[VerifiedClaim(claim="Strong", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.SENSOR_TOWER)])],
                swot=SWOTAnalysis(
                    strengths=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    weaknesses=[VerifiedClaim(claim="y", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    opportunities=[VerifiedClaim(claim="z", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    threats=[VerifiedClaim(claim="a", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
                ),
                target_audiences=[TargetAudience(segment_name="Gamers")],
                regulatory_analysis=[],  # Empty - invalid!
                regulatory_recommendation="OK",
                gap_analysis=[],
                feature_recommendations=[FeatureRecommendation(
                    feature_name="x", description="x", priority="nice_to_have",
                    rationale=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    estimated_effort="low", competitive_advantage="x"
                )],
                gtm_strategy=GoToMarketStrategy(primary_channels=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]),
                key_risks=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                mitigation_strategies=["x"],
                data_sources_used=[]
            )
    
    def test_mrd_regulatory_rec_must_reference_regions(self):
        """Regulatory recommendation should reference actual regions"""
        with pytest.raises(ValueError, match="must reference"):
            StrategicAnalysis(
                mrd_id="mrd_001",
                overall_confidence=ConfidenceLevel.HIGH,
                executive_summary="This is a long executive summary that meets minimum length requirement",
                user_intent="Build an app",
                interpreted_goal="Analyze market",
                market_overview=[VerifiedClaim(claim="Growing", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                competitor_list=[CompetitorProfile(name="Competitor")],
                competitive_moat_analysis=[VerifiedClaim(claim="Strong", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.SENSOR_TOWER)])],
                swot=SWOTAnalysis(
                    strengths=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    weaknesses=[VerifiedClaim(claim="y", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    opportunities=[VerifiedClaim(claim="z", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    threats=[VerifiedClaim(claim="a", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
                ),
                target_audiences=[TargetAudience(segment_name="Gamers")],
                regulatory_analysis=[
                    RegulatoryStatus(region="UK", legal_status="legal", confidence=ConfidenceLevel.MEDIUM, citation=Citation(source=DataSource.REGULATORY_DB))
                ],
                regulatory_recommendation="xx",  # Too short, won't reference UK
                gap_analysis=[],
                feature_recommendations=[FeatureRecommendation(
                    feature_name="x", description="x", priority="nice_to_have",
                    rationale=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    estimated_effort="low", competitive_advantage="x"
                )],
                gtm_strategy=GoToMarketStrategy(primary_channels=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]),
                key_risks=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                mitigation_strategies=["x"],
                data_sources_used=[]
            )
    
    def test_valid_complete_mrd(self):
        """Valid complete MRD should be created"""
        mrd = StrategicAnalysis(
            mrd_id="mrd_001",
            overall_confidence=ConfidenceLevel.HIGH,
            executive_summary="This is a long executive summary that meets minimum length requirement of 100 characters for validation",
            user_intent="Build a skill gaming app",
            interpreted_goal="Analyze market for skill gaming in EU",
            market_overview=[VerifiedClaim(
                claim="Growing skill gaming market",
                confidence=ConfidenceLevel.MEDIUM,
                citations=[Citation(source=DataSource.WEB_SEARCH)]
            )],
            competitor_list=[CompetitorProfile(name="Triumph")],
            competitive_moat_analysis=[VerifiedClaim(
                claim="Strong retention",
                confidence=ConfidenceLevel.HIGH,
                citations=[
                    Citation(source=DataSource.SENSOR_TOWER),
                    Citation(source=DataSource.WEB_SEARCH)
                ]
            )],
            swot=SWOTAnalysis(
                strengths=[VerifiedClaim(claim="Strong position", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                weaknesses=[VerifiedClaim(claim="Regulatory risk", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.REGULATORY_DB)])],
                opportunities=[VerifiedClaim(claim="IO segment", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                threats=[VerifiedClaim(claim="Tightening regs", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.REGULATORY_DB)])]
            ),
            target_audiences=[TargetAudience(segment_name="Young Males")],
            regulatory_analysis=[
                RegulatoryStatus(
                    region="UK",
                    legal_status="requires_license",
                    confidence=ConfidenceLevel.HIGH,
                    citation=Citation(source=DataSource.REGULATORY_DB)
                )
            ],
            regulatory_recommendation="License in UK, consider Malta for EU licensing strategy",
            gap_analysis=[],
            feature_recommendations=[
                FeatureRecommendation(
                    feature_name="IO Mode",
                    description="Agar-style games",
                    priority="should_have",
                    rationale=[VerifiedClaim(claim="Market opportunity", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    estimated_effort="medium",
                    competitive_advantage="First-mover"
                )
            ],
            gtm_strategy=GoToMarketStrategy(
                primary_channels=[VerifiedClaim(
                    claim="TikTok marketing",
                    confidence=ConfidenceLevel.HIGH,
                    citations=[
                        Citation(source=DataSource.SOCIAL_SENTIMENT),
                        Citation(source=DataSource.WEB_SEARCH)
                    ]
                )]
            ),
            key_risks=[VerifiedClaim(
                claim="Regulatory changes",
                confidence=ConfidenceLevel.MEDIUM,
                citations=[Citation(source=DataSource.REGULATORY_DB)]
            )],
            mitigation_strategies=["Diversify channels", "Build compliance team"],
            data_sources_used=["sensor_tower", "regulatory_db", "web_search", "social_sentiment"]
        )
        
        assert mrd.mrd_id == "mrd_001"
        assert mrd.overall_confidence == ConfidenceLevel.HIGH


class TestMRDValidationResult:
    """Test MRDValidationResult creation and logic"""
    
    def test_valid_approval_result(self):
        """Valid approval result can be created"""
        result = MRDValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            recommendation="approve"
        )
        assert result.is_valid
        assert result.recommendation == "approve"
    
    def test_revision_result_with_warnings(self):
        """Can create revision result with warnings"""
        result = MRDValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Some unverified claims"],
            unverified_claims_count=2,
            recommendation="revise"
        )
        assert result.is_valid
        assert len(result.warnings) == 1
        assert result.recommendation == "revise"
    
    def test_rejection_result_with_errors(self):
        """Can create rejection result with errors"""
        result = MRDValidationResult(
            is_valid=False,
            errors=["No competitor data", "No regulatory analysis"],
            warnings=[],
            recommendation="reject"
        )
        assert not result.is_valid
        assert len(result.errors) == 2
        assert result.recommendation == "reject"


class TestMRDJSONSerialization:
    """Test that MRD can be properly serialized to JSON"""
    
    def test_mrd_to_json(self):
        """MRD should serialize to valid JSON"""
        mrd = StrategicAnalysis(
            mrd_id="mrd_001",
            overall_confidence=ConfidenceLevel.HIGH,
            executive_summary="This is a long executive summary that meets minimum length requirement",
            user_intent="Build an app",
            interpreted_goal="Analyze market",
            market_overview=[VerifiedClaim(claim="Growing", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
            competitor_list=[CompetitorProfile(name="Competitor")],
            competitive_moat_analysis=[VerifiedClaim(claim="Strong", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.SENSOR_TOWER)])],
            swot=SWOTAnalysis(
                strengths=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                weaknesses=[VerifiedClaim(claim="y", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                opportunities=[VerifiedClaim(claim="z", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                threats=[VerifiedClaim(claim="a", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
            ),
            target_audiences=[TargetAudience(segment_name="Gamers")],
            regulatory_analysis=[RegulatoryStatus(
                region="UK",
                legal_status="legal",
                confidence=ConfidenceLevel.MEDIUM,
                citation=Citation(source=DataSource.REGULATORY_DB)
            )],
            regulatory_recommendation="Launch in UK market",
            gap_analysis=[],
            feature_recommendations=[FeatureRecommendation(
                feature_name="x", description="x", priority="nice_to_have",
                rationale=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                estimated_effort="low", competitive_advantage="x"
            )],
            gtm_strategy=GoToMarketStrategy(primary_channels=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]),
            key_risks=[VerifiedClaim(claim="x", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
            mitigation_strategies=["x"],
            data_sources_used=[]
        )
        
        # Should be serializable to JSON
        json_str = mrd.model_dump_json()
        assert json_str is not None
        assert len(json_str) > 0
        
        # Should be parseable back
        json_dict = json.loads(json_str)
        assert json_dict["mrd_id"] == "mrd_001"
        assert json_dict["overall_confidence"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
