"""
Test suite for Pydantic model validators.
Ensures data integrity and prevents hallucinations through validation.
"""

import pytest
from datetime import datetime

from models.core import (
    ResearchTask, ResearchPlan, VerifiedClaim, Citation, DataSource,
    ConfidenceLevel, ResearchAggregate, CompetitorProfile, RegulatoryStatus
)
from models.mrd import StrategicAnalysis, SWOTAnalysis, MRDValidationResult


class TestResearchTaskValidation:
    """Test ResearchTask validators"""
    
    def test_valid_research_task(self):
        """Valid research task should be created successfully"""
        task = ResearchTask(
            task_id="test_task",
            task_type="competitor",
            query="Analyze market leaders in skill gaming",
            target_entities=["Triumph", "Skillz"],
            required_tools=["search_sensor_tower"],
            success_criteria="Obtain revenue and retention data"
        )
        assert task.task_id == "test_task"
        assert len(task.required_tools) == 1
    
    def test_research_task_empty_required_tools_fails(self):
        """Task with no required tools should fail"""
        with pytest.raises(ValueError, match="At least one required tool"):
            ResearchTask(
                task_id="test_task",
                task_type="competitor",
                query="Analyze market leaders in skill gaming",
                target_entities=["Triumph"],
                required_tools=[],  # Invalid!
                success_criteria="Obtain data"
            )
    
    def test_research_task_short_query_fails(self):
        """Task with query < 10 chars should fail"""
        with pytest.raises(ValueError):
            ResearchTask(
                task_id="test_task",
                task_type="competitor",
                query="Short",  # Too short
                target_entities=["Triumph"],
                required_tools=["search_sensor_tower"],
                success_criteria="Data"
            )
    
    def test_research_task_with_fallback_tools(self):
        """Task should support fallback tools"""
        task = ResearchTask(
            task_id="test_task",
            task_type="competitor",
            query="Analyze market leaders in skill gaming",
            target_entities=["Triumph"],
            required_tools=["search_sensor_tower"],
            fallback_tools=["web_search"],
            success_criteria="Obtain data"
        )
        assert len(task.fallback_tools) == 1


class TestResearchPlanValidation:
    """Test ResearchPlan validators"""
    
    def test_valid_research_plan(self):
        """Valid research plan should be created"""
        plan = ResearchPlan(
            plan_id="plan_001",
            user_intent="Build a skill gaming app",
            interpreted_goal="Analyze market opportunities in skill gaming",
            tasks=[
                ResearchTask(
                    task_id="task_1",
                    task_type="competitor",
                    query="Analyze competitor apps in skill gaming space",
                    required_tools=["search_sensor_tower"],
                    success_criteria="Get competitor data"
                )
            ],
            estimated_duration_minutes=30
        )
        assert plan.plan_id == "plan_001"
        assert not plan.human_approved  # Should default to False
    
    def test_research_plan_duplicate_task_ids_fails(self):
        """Plan with duplicate task IDs should fail"""
        with pytest.raises(ValueError, match="Duplicate task IDs"):
            ResearchPlan(
                plan_id="plan_001",
                user_intent="Build an app",
                interpreted_goal="Goal",
                tasks=[
                    ResearchTask(
                        task_id="same_id",
                        task_type="competitor",
                        query="Query that is long enough here",
                        required_tools=["search_sensor_tower"],
                        success_criteria="Data"
                    ),
                    ResearchTask(
                        task_id="same_id",  # Duplicate!
                        task_type="market",
                        query="Another query that is long enough here",
                        required_tools=["web_search"],
                        success_criteria="Data"
                    )
                ],
                estimated_duration_minutes=30
            )
    
    def test_research_plan_invalid_duration_fails(self):
        """Plan with invalid duration should fail"""
        with pytest.raises(ValueError, match="Duration"):
            ResearchPlan(
                plan_id="plan_001",
                user_intent="Build an app",
                interpreted_goal="Goal",
                tasks=[
                    ResearchTask(
                        task_id="task_1",
                        task_type="competitor",
                        query="Query that is long enough here",
                        required_tools=["search_sensor_tower"],
                        success_criteria="Data"
                    )
                ],
                estimated_duration_minutes=2000  # Too long (>1440)
            )
    
    def test_research_plan_with_human_approval(self):
        """Plan can track human approval"""
        plan = ResearchPlan(
            plan_id="plan_001",
            user_intent="Build an app",
            interpreted_goal="Goal",
            tasks=[
                ResearchTask(
                    task_id="task_1",
                    task_type="competitor",
                    query="Query that is long enough here",
                    required_tools=["search_sensor_tower"],
                    success_criteria="Data"
                )
            ],
            estimated_duration_minutes=30,
            human_approved=True,
            approval_timestamp=datetime.utcnow(),
            approval_notes="Plan looks good"
        )
        assert plan.human_approved
        assert plan.approval_notes is not None


class TestVerifiedClaimValidation:
    """Test VerifiedClaim validators"""
    
    def test_valid_claim_with_single_citation(self):
        """Claim with 1 citation should work for non-HIGH confidence"""
        claim = VerifiedClaim(
            claim="Triumph has 35% 30-day retention",
            confidence=ConfidenceLevel.MEDIUM,
            citations=[
                Citation(source=DataSource.SENSOR_TOWER, url="https://example.com")
            ]
        )
        assert len(claim.citations) == 1
    
    def test_high_confidence_requires_multiple_citations(self):
        """HIGH confidence claims need at least 2 citations"""
        with pytest.raises(ValueError, match="HIGH confidence claims require at least 2"):
            VerifiedClaim(
                claim="Triumph has 35% 30-day retention",
                confidence=ConfidenceLevel.HIGH,  # High confidence!
                citations=[
                    Citation(source=DataSource.SENSOR_TOWER)  # Only 1 citation
                ]
            )
    
    def test_high_confidence_with_multiple_citations(self):
        """HIGH confidence with 2+ citations should work"""
        claim = VerifiedClaim(
            claim="Triumph has 35% 30-day retention",
            confidence=ConfidenceLevel.HIGH,
            citations=[
                Citation(source=DataSource.SENSOR_TOWER),
                Citation(source=DataSource.WEB_SEARCH)
            ]
        )
        assert len(claim.citations) == 2
    
    def test_claim_min_length_validation(self):
        """Claim must be at least 5 characters"""
        with pytest.raises(ValueError):
            VerifiedClaim(
                claim="Bad",  # Too short
                confidence=ConfidenceLevel.MEDIUM,
                citations=[Citation(source=DataSource.SENSOR_TOWER)]
            )
    
    def test_claim_requires_at_least_one_citation(self):
        """Claim must have at least one citation"""
        with pytest.raises(ValueError):
            VerifiedClaim(
                claim="Triumph is winning",
                confidence=ConfidenceLevel.MEDIUM,
                citations=[]  # No citations!
            )


class TestResearchAggregateCompleteness:
    """Test ResearchAggregate completeness scoring"""
    
    def test_empty_aggregate_is_zero_percent(self):
        """Empty aggregate should score 0%"""
        agg = ResearchAggregate(plan_id="test")
        completeness = agg.calculate_completeness()
        assert completeness == 0.0
    
    def test_partial_data_scores_partially(self):
        """Aggregate with partial data should score between 0-100%"""
        agg = ResearchAggregate(
            plan_id="test",
            competitors=[
                CompetitorProfile(name="Triumph"),
                CompetitorProfile(name="Skillz")
            ]
        )
        completeness = agg.calculate_completeness()
        assert 0 < completeness < 1.0
    
    def test_completeness_by_dimension(self):
        """Completeness tracking should break down by dimension"""
        agg = ResearchAggregate(
            plan_id="test",
            competitors=[CompetitorProfile(name="Triumph")],
            sentiment_analyses=[]
        )
        agg.calculate_completeness()
        
        assert 'competitors' in agg.completeness_by_dimension
        assert 'sentiment' in agg.completeness_by_dimension
        assert agg.completeness_by_dimension['competitors'] > 0
        assert agg.completeness_by_dimension['sentiment'] == 0


class TestMRDValidationResult:
    """Test MRDValidationResult validators"""
    
    def test_valid_validation_result(self):
        """Valid validation result should be created"""
        result = MRDValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            recommendation="approve"
        )
        assert result.is_valid
        assert result.recommendation == "approve"
    
    def test_cannot_approve_with_errors(self):
        """Cannot approve when errors exist"""
        with pytest.raises(ValueError, match="Cannot approve MRD with errors"):
            MRDValidationResult(
                is_valid=False,
                errors=["Missing competitor data"],
                recommendation="approve"  # Invalid combo!
            )
    
    def test_cannot_approve_invalid_mrd(self):
        """Cannot approve invalid MRD"""
        with pytest.raises(ValueError, match="Cannot approve invalid MRD"):
            MRDValidationResult(
                is_valid=False,
                errors=[],
                recommendation="approve"  # Invalid combo!
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
