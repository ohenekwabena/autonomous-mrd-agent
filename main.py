"""
Main entry point demonstrating the complete pipeline.
"""

import asyncio
from typing import Optional
import uuid

from models.core import (
    AgentState, ResearchPlan, ResearchTask, ResearchAggregate,
    MarketData, VerifiedClaim, Citation, DataSource, ConfidenceLevel,
    GapAnalysisItem
)
from models.mrd import (
    StrategicAnalysis, SWOTAnalysis, FeatureRecommendation,
    TargetAudience, GoToMarketStrategy, MRDValidationResult
)
from orchestration.state_machine import Orchestrator, OrchestratorContext
from agents.research_agents import (
    MockToolkit, CompetitorAnalysisAgent, SentimentAnalysisAgent,
    RegulatoryAnalysisAgent
)


# ============================================================================
# STATE HANDLERS
# ============================================================================

async def planning_handler(ctx:  OrchestratorContext) -> OrchestratorContext:
    """
    Generate a research plan from user prompt.
    In production:  This would use an LLM to interpret the prompt. 
    """
    print(f"[PLANNING] Interpreting prompt: {ctx.user_prompt[: 50]}...")
    
    # Simulate LLM interpretation
    ctx.research_plan = ResearchPlan(
        plan_id=str(uuid.uuid4()),
        user_intent=ctx.user_prompt,
        interpreted_goal="Build a skill-based gaming app for European market, similar to Triumph",
        tasks=[
            ResearchTask(
                task_id="competitor_analysis",
                task_type="competitor",
                query="Analyze Triumph and Skillz apps for market positioning",
                target_entities=["Triumph", "Skillz"],
                required_tools=["search_sensor_tower"],
                success_criteria="Obtain MAU, revenue, and retention data for both apps"
            ),
            ResearchTask(
                task_id="sentiment_analysis",
                task_type="sentiment",
                query="Analyze TikTok sentiment for Triumph app marketing",
                target_entities=["Triumph"],
                required_tools=["analyze_sentiment"],
                success_criteria="Obtain sentiment ratios and influencer data"
            ),
            ResearchTask(
                task_id="regulatory_check_uk",
                task_type="regulatory",
                query="Check gambling regulations in UK",
                target_entities=["UK"],
                required_tools=["check_regulatory_compliance"],
                success_criteria="Determine legal status and license requirements"
            ),
            ResearchTask(
                task_id="regulatory_check_eu",
                task_type="regulatory",
                query="Check gambling regulations in EU",
                target_entities=["EU"],
                required_tools=["check_regulatory_compliance"],
                success_criteria="Determine legal status across EU member states"
            ),
            ResearchTask(
                task_id="gap_analysis",
                task_type="gap_analysis",
                query="Identify IO games not offered by Triumph",
                target_entities=["Triumph"],
                required_tools=["web_search"],
                success_criteria="List at least 3 potential game opportunities"
            )
        ],
        estimated_duration_minutes=5
    )
    
    print(f"[PLANNING] Generated plan with {len(ctx. research_plan.tasks)} tasks")
    
    # In production: Wait for human approval here
    # For demo, auto-approve
    return ctx


async def research_handler(ctx:  OrchestratorContext) -> OrchestratorContext: 
    """
    Execute research tasks using specialized agents.
    """
    print("[RESEARCH] Starting research phase...")
    
    toolkit = MockToolkit()
    
    # Initialize aggregate if not exists
    if ctx.research_aggregate is None:
        ctx.research_aggregate = ResearchAggregate(
            plan_id=ctx. research_plan.plan_id
        )
    
    # Execute each task
    for task in ctx.research_plan.tasks:
        if task.task_id in [t for t in ctx.research_aggregate.failed_tasks]:
            continue  # Skip already failed tasks
        
        print(f"[RESEARCH] Executing task: {task.task_id}")
        
        try:
            if task.task_type == "competitor": 
                agent = CompetitorAnalysisAgent(toolkit)
                competitors = await agent.run(task)
                ctx.research_aggregate.competitors.extend(competitors)
                
            elif task.task_type == "sentiment":
                agent = SentimentAnalysisAgent(toolkit)
                for entity in task.target_entities:
                    result = await agent.analyze_platform("tiktok", entity)
                    if result:
                        ctx.research_aggregate. sentiment_analyses.append(result)
                        
            elif task.task_type == "regulatory": 
                agent = RegulatoryAnalysisAgent(toolkit)
                for region in task.target_entities:
                    result = await agent.check_region(region)
                    if result:
                        ctx.research_aggregate. regulatory_statuses.append(result)
                        
            elif task.task_type == "gap_analysis":
                # Mock gap analysis results
                ctx.research_aggregate.gap_analysis. append(
                    GapAnalysisItem(
                        opportunity="Agar.io-style games",
                        current_market_state="Popular IO game not offered by Triumph",
                        potential_value="high",
                        implementation_complexity="medium",
                        supporting_evidence=[
                            VerifiedClaim(
                                claim="Agar.io has 50M+ downloads on mobile",
                                confidence=ConfidenceLevel. MEDIUM,
                                citations=[Citation(source=DataSource.WEB_SEARCH)]
                            )
                        ]
                    )
                )
                
        except Exception as e: 
            ctx.research_aggregate.failed_tasks.append(task.task_id)
            ctx.errors.append(f"Task {task.task_id} failed: {str(e)}")
    
    # Calculate completeness
    completeness = ctx.research_aggregate.calculate_completeness()
    print(f"[RESEARCH] Data completeness: {completeness:.1%}")
    
    return ctx


async def synthesis_handler(ctx:  OrchestratorContext) -> OrchestratorContext: 
    """
    Synthesize research data into structured MRD. 
    In production: This would use an LLM with strict output parsing.
    """
    print("[SYNTHESIS] Generating MRD...")
    
    agg = ctx.research_aggregate
    
    # Build SWOT from collected data
    swot = SWOTAnalysis(
        strengths=[
            VerifiedClaim(
                claim="Strong influencer presence on TikTok with 1M+ follower creators",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource. SOCIAL_SENTIMENT)]
            )
        ],
        weaknesses=[
            VerifiedClaim(
                claim="Regulatory complexity in EU market requires per-country licensing",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource.REGULATORY_DB)]
            )
        ],
        opportunities=[
            VerifiedClaim(
                claim="IO game category underserved - potential for Agar.io-style games",
                confidence=ConfidenceLevel.MEDIUM,
                citations=[Citation(source=DataSource.WEB_SEARCH)]
            )
        ],
        threats=[
            VerifiedClaim(
                claim="UK Gambling Commission 2024 white paper tightening advertising rules",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource.REGULATORY_DB)]
            )
        ]
    )
    
    # Generate MRD
    ctx.mrd = StrategicAnalysis(
        mrd_id=str(uuid.uuid4()),
        overall_confidence=ConfidenceLevel. MEDIUM,
        executive_summary=(
            "Analysis indicates strong potential for a skill-based gaming app in the European market.  "
            "Triumph's success demonstrates product-market fit, with 35% 30-day retention vs Skillz's 18%.  "
            "Key success factors include TikTok influencer marketing and careful regulatory navigation.  "
            "UK market requires Remote Gambling License; EU strategy should prioritize Malta/Gibraltar jurisdictions.  "
            "Gap analysis reveals opportunity in IO-style games not currently offered by competitors."
        ),
        user_intent=ctx.user_prompt,
        interpreted_goal=ctx.research_plan.interpreted_goal,
        market_overview=[
            VerifiedClaim(
                claim="Skill gaming market growing as Skillz public company struggles with 18% retention",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource.SENSOR_TOWER)]
            )
        ],
        competitor_list=agg.competitors,
        competitive_moat_analysis=[
            VerifiedClaim(
                claim="Triumph's 35% retention vs Skillz 18% indicates superior game selection/UX",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource. SENSOR_TOWER)]
            )
        ],
        swot=swot,
        target_audiences=[
            TargetAudience(
                segment_name="Young Male Gamers",
                demographics={"age_range": "18-34", "gender": "male", "income": "middle"},
                psychographics=["competitive", "mobile-first", "social media active"],
                acquisition_channels=[
                    VerifiedClaim(
                        claim="TikTok primary channel with 62% positive sentiment",
                        confidence=ConfidenceLevel.HIGH,
                        citations=[Citation(source=DataSource.SOCIAL_SENTIMENT)]
                    )
                ]
            )
        ],
        regulatory_analysis=agg.regulatory_statuses,
        regulatory_recommendation=(
            "Start with UK (clear licensing path) and Malta-licensed EU operations. "
            "Avoid Germany/Netherlands initially due to restrictive regulations."
        ),
        gap_analysis=agg.gap_analysis,
        feature_recommendations=[
            FeatureRecommendation(
                feature_name="IO Game Mode",
                description="Agar.io-style multiplayer skill game",
                priority="should_have",
                rationale=[
                    VerifiedClaim(
                        claim="Genre has 50M+ mobile downloads, not offered by Triumph",
                        confidence=ConfidenceLevel. MEDIUM,
                        citations=[Citation(source=DataSource. WEB_SEARCH)]
                    )
                ],
                estimated_effort="medium",
                competitive_advantage="First-mover in skill gaming IO genre"
            )
        ],
        gtm_strategy=GoToMarketStrategy(
            primary_channels=[
                VerifiedClaim(
                    claim="TikTok influencer marketing - 1M+ follower gaming creators active",
                    confidence=ConfidenceLevel.HIGH,
                    citations=[Citation(source=DataSource.SOCIAL_SENTIMENT)]
                )
            ],
            influencer_strategy="Partner with mid-tier gaming influencers (100K-1M followers)",
            geographic_rollout=["UK", "Malta", "Ireland"],
            regulatory_considerations=["UK requires Gambling Commission license", "Malta offers favorable licensing"]
        ),
        key_risks=[
            VerifiedClaim(
                claim="Regulatory changes in 2024 may restrict marketing channels",
                confidence=ConfidenceLevel.HIGH,
                citations=[Citation(source=DataSource. REGULATORY_DB)]
            )
        ],
        mitigation_strategies=[
            "Diversify acquisition beyond TikTok",
            "Build compliance team early",
            "Consider skill-only games for less regulated markets"
        ],
        data_sources_used=[s.value for s in DataSource if s != DataSource.INFERRED],
        claims_requiring_verification=[],
        research_gaps=agg.failed_tasks
    )
    
    print(f"[SYNTHESIS] MRD generated with {len(ctx.mrd.feature_recommendations)} recommendations")
    
    return ctx


async def validation_handler(ctx:  OrchestratorContext) -> OrchestratorContext: 
    """
    Validate the generated MRD. 
    """
    print("[VALIDATION] Validating MRD...")
    
    errors = []
    warnings = []
    unverified_count = 0
    low_confidence_count = 0
    
    # Check for unverified claims
    def check_claims(claims:  list[VerifiedClaim]):
        nonlocal unverified_count, low_confidence_count
        for claim in claims:
            if claim.confidence == ConfidenceLevel. UNVERIFIED: 
                unverified_count += 1
            elif claim.confidence == ConfidenceLevel.LOW:
                low_confidence_count += 1
    
    check_claims(ctx.mrd.market_overview)
    check_claims(ctx.mrd.swot. strengths)
    check_claims(ctx.mrd.swot.weaknesses)
    check_claims(ctx.mrd.swot.opportunities)
    check_claims(ctx.mrd.swot. threats)
    
    # Validation rules
    if not ctx.mrd. competitor_list:
        errors.append("No competitor analysis included")
    
    if not ctx.mrd.regulatory_analysis:
        errors. append("No regulatory analysis included")
    
    if unverified_count > 5:
        warnings. append(f"{unverified_count} claims require verification")
    
    if len(ctx.mrd.executive_summary) < 100:
        warnings.append("Executive summary may be too brief")
    
    # Determine recommendation
    if errors: 
        recommendation = "reject"
    elif unverified_count > 3 or low_confidence_count > 5:
        recommendation = "revise"
    else:
        recommendation = "approve"
    
    ctx.validation_result = MRDValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        unverified_claims_count=unverified_count,
        low_confidence_claims_count=low_confidence_count,
        recommendation=recommendation
    )
    
    print(f"[VALIDATION] Result: {recommendation} ({len(errors)} errors, {len(warnings)} warnings)")
    
    return ctx


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run the autonomous MRD agent"""
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Register handlers for each state
    orchestrator.register_handler(AgentState.PLANNING, planning_handler)
    orchestrator. register_handler(AgentState.RESEARCH, research_handler)
    orchestrator. register_handler(AgentState.SYNTHESIS, synthesis_handler)
    orchestrator. register_handler(AgentState.VALIDATION, validation_handler)
    
    # User prompt
    user_prompt = """
    I want to build a skill-based gambling app targeting young men, 
    similar to Triumph but for the European market.  
    Help me understand the competitive landscape, regulatory requirements,
    and identify opportunities in the IO gaming space.
    """
    
    # Run the pipeline
    print("=" * 60)
    print("AUTONOMOUS MRD AGENT")
    print("=" * 60)
    
    result = await orchestrator. run(user_prompt)
    
    # Output results
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Final State: {result.current_state}")
    print(f"Total Transitions: {len(result.transitions)}")
    print(f"Errors Encountered: {len(result.errors)}")
    
    if result.mrd: 
        print(f"\nMRD Generated:")
        print(f"  - Competitors Analyzed: {len(result.mrd.competitor_list)}")
        print(f"  - Regulatory Regions: {len(result.mrd.regulatory_analysis)}")
        print(f"  - Feature Recommendations: {len(result.mrd.feature_recommendations)}")
        print(f"  - Overall Confidence: {result. mrd.overall_confidence}")
        
        # Export as JSON (what goes to DB)
        print("\n[OUTPUT] MRD JSON available for database storage")
        # print(result.mrd.model_dump_json(indent=2))
    
    return result


if __name__ == "__main__": 
    asyncio. run(main())