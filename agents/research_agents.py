"""
Research agents that interface with tools and handle failures gracefully.
Implements fallback ladders for tool failures.
"""

from __future__ import annotations
import asyncio
from typing import Optional, Callable, Awaitable
from datetime import datetime

from models.core import (
    ResearchTask, ResearchAggregate, CompetitorProfile, MarketData,
    SentimentAnalysis, RegulatoryStatus, GapAnalysisItem,
    VerifiedClaim, Citation, DataSource, ConfidenceLevel, TaskProvenance
)
from tools.mock_toolkit import ToolResponse, MockToolkit, FallbackLadder


# ============================================================================
# FALLBACK LADDER DEFINITIONS
# ============================================================================


# ============================================================================
# RESEARCH AGENT BASE WITH RETRY & FALLBACK
# ============================================================================

class BaseResearchAgent: 
    """Base class for all research agents with retry logic and fallback ladders"""
    
    def __init__(self, toolkit: MockToolkit):
        self.toolkit = toolkit
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def execute_with_retry_and_fallback(
        self, 
        task: ResearchTask,
        primary_tool: str,
        tool_func: Callable,
        *args,
        **kwargs
    ) -> tuple[ToolResponse, list[TaskProvenance]]:
        """
        Execute a tool with retry logic and fallback ladder.
        Returns (final_response, execution_history)
        """
        provenance_trail = []
        attempted_tools = [primary_tool]
        
        # Try primary tool first
        last_response = await self._retry_tool_with_attempts(
            tool_func, primary_tool, args, kwargs, task.max_retries, provenance_trail, task.task_id
        )
        
        if last_response.success:
            return last_response, provenance_trail
        
        # Primary tool failed - try fallback ladder
        fallback_tools = FallbackLadder.get_fallback_tools(primary_tool)
        
        for fallback_tool in fallback_tools:
            attempted_tools.append(fallback_tool)
            print(f"[FALLBACK] Primary tool {primary_tool} failed. Trying fallback: {fallback_tool}")
            
            # Get tool function for fallback
            fallback_func = getattr(self.toolkit, fallback_tool, None)
            if not fallback_func:
                continue
            
            last_response = await self._retry_tool_with_attempts(
                fallback_func, fallback_tool, args, kwargs, 2, provenance_trail, task.task_id
            )
            
            if last_response.success:
                print(f"[SUCCESS] Fallback tool {fallback_tool} succeeded")
                return last_response, provenance_trail
        
        # All tools exhausted
        error_msg = f"All tools exhausted ({attempted_tools}). Last error: {last_response.error_message}"
        final_response = ToolResponse(
            success=False,
            tool_name="none",
            error_message=error_msg
        )
        
        return final_response, provenance_trail
    
    async def _retry_tool_with_attempts(
        self,
        tool_func: Callable,
        tool_name: str,
        args,
        kwargs,
        max_attempts: int,
        provenance_trail: list[TaskProvenance],
        task_id: str
    ) -> ToolResponse:
        """Execute tool with retry attempts and track provenance"""
        last_response = None
        
        for attempt in range(max_attempts):
            try:
                response = await tool_func(*args, **kwargs)
                response.tool_name = tool_name
                
                # Record provenance
                prov = TaskProvenance(
                    task_id=task_id,
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    success=response.success,
                    data_hash=response.get_hash(),
                    error_message=response.error_message if not response.success else None
                )
                provenance_trail.append(prov)
                
                if response.success:
                    return response
                last_response = response
                
            except Exception as e: 
                prov = TaskProvenance(
                    task_id=task_id,
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    success=False,
                    error_message=str(e)
                )
                provenance_trail.append(prov)
                last_response = ToolResponse(
                    success=False,
                    tool_name=tool_name,
                    error_message=str(e)
                )
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return last_response or ToolResponse(
            success=False,
            tool_name=tool_name,
            error_message=f"Failed after {max_attempts} attempts"
        )
    
    def create_citation(self, source: DataSource, response: ToolResponse, url: str = None) -> Citation:
        """Create a citation from a tool response"""
        return Citation(
            source=source,
            url=url,
            retrieved_at=datetime.utcnow(),
            raw_data_hash=response.get_hash()
        )


# ============================================================================
# SPECIALIZED RESEARCH AGENTS
# ============================================================================

class CompetitorAnalysisAgent(BaseResearchAgent):
    """Agent specialized in competitor research with fallback support"""
    
    async def analyze_competitor(self, app_name: str, task: ResearchTask) -> Optional[CompetitorProfile]: 
        """Analyze a single competitor with fallback"""
        response, provenance = await self.execute_with_retry_and_fallback(
            task,
            "search_sensor_tower",
            self.toolkit.search_sensor_tower,
            app_name
        )
        
        if not response.success:
            return None
        
        data = response.data
        citation = self.create_citation(DataSource.SENSOR_TOWER, response)
        
        return CompetitorProfile(
            name=data. get("app_name", app_name),
            monthly_active_users=data. get("monthly_downloads"),
            revenue_estimate=data. get("revenue_estimate"),
            key_features=[],  # Would need additional research
            target_demographics=data.get("top_countries", []),
            strengths=[
                VerifiedClaim(
                    claim=f"Strong user retention at {data. get('user_retention_30d', 0)*100:.0f}%",
                    confidence=ConfidenceLevel. HIGH,
                    citations=[citation]
                )
            ] if data.get("user_retention_30d", 0) > 0.25 else [],
            weaknesses=[]
        )
    
    async def run(self, task: ResearchTask) -> list[CompetitorProfile]: 
        """Execute competitor analysis task"""
        results = []
        for entity in task.target_entities:
            profile = await self.analyze_competitor(entity, task)
            if profile:
                results.append(profile)
        return results


class SentimentAnalysisAgent(BaseResearchAgent):
    """Agent specialized in social sentiment analysis"""
    
    async def analyze_platform(
        self, 
        platform: str, 
        query:  str,
        task: ResearchTask
    ) -> Optional[SentimentAnalysis]:
        """Analyze sentiment on a specific platform with fallback"""
        response, provenance = await self.execute_with_retry_and_fallback(
            task,
            "analyze_sentiment",
            self.toolkit.analyze_sentiment,
            platform,
            query
        )
        
        if not response.success:
            return None
        
        data = response.data
        sentiment = data.get("sentiment", {})
        
        return SentimentAnalysis(
            platform=platform,
            sample_size=data. get("sample_size", 0),
            positive_ratio=sentiment.get("positive", 0),
            negative_ratio=sentiment.get("negative", 0),
            neutral_ratio=sentiment.get("neutral", 0),
            top_positive_themes=data.get("top_hashtags", []),
            influencer_mentions=data.get("influencer_mentions", []),
            citation=self.create_citation(DataSource. SOCIAL_SENTIMENT, response)
        )


class RegulatoryAnalysisAgent(BaseResearchAgent):
    """Agent specialized in regulatory compliance research"""
    
    async def check_region(self, region: str, task: ResearchTask) -> Optional[RegulatoryStatus]:
        """Check regulatory status for a region with fallback"""
        response, provenance = await self.execute_with_retry_and_fallback(
            task,
            "check_regulatory_compliance",
            self.toolkit.check_regulatory_compliance,
            region
        )
        
        if not response. success:
            return RegulatoryStatus(
                region=region,
                legal_status="unknown",
                confidence=ConfidenceLevel.UNVERIFIED,
                citation=Citation(source=DataSource. MANUAL_INPUT)
            )
        
        data = response.data
        citation = self.create_citation(DataSource. REGULATORY_DB, response)
        
        # Map status string to enum
        status_map = {
            "requires_license": "requires_license",
            "legal":  "legal",
            "illegal": "illegal",
            "varies_by_country": "gray_area",
            "varies_by_state": "gray_area"
        }
        
        return RegulatoryStatus(
            region=region,
            legal_status=status_map. get(data.get("status", "unknown"), "unknown"),
            license_requirements=data.get("license_types", []),
            restrictions=data.get("restrictions", []),
            recent_changes=[
                VerifiedClaim(
                    claim=data.get("recent_changes", ""),
                    confidence=ConfidenceLevel. MEDIUM,
                    citations=[citation]
                )
            ] if data.get("recent_changes") else [],
            confidence=ConfidenceLevel.HIGH if response.success else ConfidenceLevel.LOW,
            citation=citation
        )
