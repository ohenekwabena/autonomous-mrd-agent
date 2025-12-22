"""
Main entry point demonstrating the complete pipeline with Gemini integration.
Now uses Gemini for research data generation instead of mocks.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
import uuid
import json

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
from adapters.gemini_client import GeminiClient
from adapters.research_generator import ResearchGenerator
from handlers.planning import planning_handler
from handlers.synthesis import synthesis_handler


# ============================================================================
# SERVICES
# ============================================================================

class Services:
    """Centralized services for handlers."""
    def __init__(self):
        self.llm = GeminiClient()


def inject_services(handler):
    """Decorator to inject services into context."""
    async def wrapped(ctx):
        if not ctx.services:
            ctx.services = Services()
        return await handler(ctx)
    return wrapped


# ============================================================================
# STATE HANDLERS (using Gemini for planning and synthesis)
# ============================================================================

async def planning_handler_main(ctx:  OrchestratorContext) -> OrchestratorContext:
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
    # For demo, auto-approve to allow pipeline to progress
    ctx.plan_approved_by_human = True
    ctx.plan_approval_timestamp = datetime.now(timezone.utc)
    return ctx


async def research_handler(ctx:  OrchestratorContext) -> OrchestratorContext: 
    """
    Execute research tasks using Gemini to generate realistic research data.
    """
    print("[RESEARCH] Starting research phase (using Gemini for data generation)...")
    
    # Use Gemini to generate research data
    generator = ResearchGenerator(ctx.services.llm)
    
    # Initialize aggregate if not exists
    if ctx.research_aggregate is None:
        ctx.research_aggregate = ResearchAggregate(
            plan_id=ctx.research_plan.plan_id
        )
    
    # Group tasks by type
    tasks_by_type = {}
    for task in ctx.research_plan.tasks:
        task_type = task.task_type
        if task_type not in tasks_by_type:
            tasks_by_type[task_type] = []
        tasks_by_type[task_type].append(task)
    
    # Execute each task type
    async def execute_task(task: ResearchTask):
        """Execute a single research task using Gemini"""
        if task.task_id in ctx.research_aggregate.failed_tasks:
            return (task.task_id, None, "Already failed")
        
        print(f"[RESEARCH] Executing task: {task.task_id} ({task.task_type})")
        
        try:
            if task.task_type == "competitor":
                result = await generator.generate_competitor_research(task.target_entities)
                if result["success"]:
                    # Convert to CompetitorProfile objects
                    competitors = []
                    for comp_data in result["data"]:
                        from models.core import CompetitorProfile
                        cp = CompetitorProfile(
                            name=comp_data.get("app_name", "Unknown"),
                            monthly_active_users=comp_data.get("monthly_downloads"),
                            revenue_estimate=comp_data.get("revenue_estimate"),
                            key_features=comp_data.get("key_features", []),
                            target_demographics=comp_data.get("top_countries", []),
                            marketing_channels=comp_data.get("marketing_channels", [])
                        )
                        competitors.append(cp)
                    return (task.task_id, ("competitor", competitors), None)
                else:
                    return (task.task_id, None, result.get("error", "Unknown error"))
                    
            elif task.task_type == "regulatory":
                result = await generator.generate_regulatory_research(task.target_entities)
                if result["success"]:
                    from models.core import RegulatoryStatus
                    statuses = []
                    for reg_data in result["data"]:
                        status_map = {
                            "legal": "legal",
                            "illegal": "illegal",
                            "gray_area": "gray_area",
                            "requires_license": "requires_license"
                        }
                        rs = RegulatoryStatus(
                            region=reg_data.get("region", "Unknown"),
                            legal_status=status_map.get(reg_data.get("status", "unknown"), "unknown"),
                            license_requirements=reg_data.get("license_types", []),
                            restrictions=reg_data.get("restrictions", []),
                            confidence=ConfidenceLevel.MEDIUM,
                            citation=Citation(source=DataSource.REGULATORY_DB)
                        )
                        statuses.append(rs)
                    return (task.task_id, ("regulatory", statuses), None)
                else:
                    return (task.task_id, None, result.get("error", "Unknown error"))
                    
            elif task.task_type == "sentiment":
                # Use first entity as the query
                query = task.target_entities[0] if task.target_entities else task.query
                result = await generator.generate_sentiment_analysis("tiktok", query)
                if result["success"]:
                    from models.core import SentimentAnalysis
                    data = result["data"]
                    # Handle sentiment being a dict or list
                    sentiment = data.get("sentiment", {})
                    if isinstance(sentiment, list):
                        sentiment = {}  # fallback
                    sa = SentimentAnalysis(
                        platform=data.get("platform", "tiktok"),
                        sample_size=data.get("sample_size", 1000),
                        positive_ratio=float(sentiment.get("positive", 0.5)),
                        negative_ratio=float(sentiment.get("negative", 0.2)),
                        neutral_ratio=float(sentiment.get("neutral", 0.3)),
                        top_positive_themes=data.get("top_themes", {}).get("positive", []) if isinstance(data.get("top_themes"), dict) else [],
                        top_negative_themes=data.get("top_themes", {}).get("negative", []) if isinstance(data.get("top_themes"), dict) else [],
                        citation=Citation(source=DataSource.SOCIAL_SENTIMENT)
                    )
                    return (task.task_id, ("sentiment", [sa]), None)
                else:
                    return (task.task_id, None, result.get("error", "Unknown error"))
                    
            elif task.task_type == "market":
                result = await generator.generate_market_research(task.query)
                if result["success"]:
                    data = result["data"]
                    market_data = MarketData(
                        market_size_usd=data.get("market_size_usd"),
                        growth_rate_percent=data.get("growth_rate_percent"),
                        key_trends=[
                            VerifiedClaim(
                                claim=trend,
                                confidence=ConfidenceLevel.MEDIUM,
                                citations=[Citation(source=DataSource.WEB_SEARCH)]
                            ) for trend in data.get("key_trends", [])
                        ]
                    )
                    return (task.task_id, ("market", market_data), None)
                else:
                    return (task.task_id, None, result.get("error", "Unknown error"))
                    
            elif task.task_type == "gap_analysis":
                gap_item = GapAnalysisItem(
                    opportunity="Advanced AI-powered game matching",
                    current_market_state="Most apps use basic matching algorithms",
                    potential_value="high",
                    implementation_complexity="medium",
                    supporting_evidence=[
                        VerifiedClaim(
                            claim="AI matching could increase retention by 15-20%",
                            confidence=ConfidenceLevel.MEDIUM,
                            citations=[Citation(source=DataSource.WEB_SEARCH)]
                        )
                    ]
                )
                return (task.task_id, ("gap_analysis", [gap_item]), None)
            
            return (task.task_id, None, "Unknown task type")
                
        except Exception as e:
            print(f"[RESEARCH ERROR] Task {task.task_id} failed: {str(e)}")
            return (task.task_id, None, str(e))
    
    # Execute all tasks concurrently
    task_coroutines = [execute_task(task) for task in ctx.research_plan.tasks]
    results = await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            ctx.errors.append(f"Unexpected error: {str(result)}")
            continue
            
        task_id, data, error = result
        
        if error:
            ctx.research_aggregate.failed_tasks.append(task_id)
            ctx.errors.append(f"Task {task_id} failed: {error}")
            continue
        
        if data:
            task_type, task_results = data
            if task_type == "competitor":
                ctx.research_aggregate.competitors.extend(task_results)
            elif task_type == "sentiment":
                ctx.research_aggregate.sentiment_analyses.extend(task_results)
            elif task_type == "regulatory":
                ctx.research_aggregate.regulatory_statuses.extend(task_results)
            elif task_type == "market":
                ctx.research_aggregate.market_data = task_results
            elif task_type == "gap_analysis":
                ctx.research_aggregate.gap_analysis.extend(task_results)
    
    # Calculate completeness
    completeness = ctx.research_aggregate.calculate_completeness()
    successful_tasks = len(ctx.research_plan.tasks) - len(ctx.research_aggregate.failed_tasks)
    print(f"[RESEARCH] Data completeness: {completeness:.1%}")
    print(f"[RESEARCH] Completed {successful_tasks}/{len(ctx.research_plan.tasks)} tasks")
    
    return ctx


async def synthesis_handler_main(ctx:  OrchestratorContext) -> OrchestratorContext: 
    """
    Fallback synthesis handler (non-Gemini mock).
    Gemini version is injected in main().
    """
    print("[SYNTHESIS] Generating MRD (fallback mock)...")
    
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
    """Run the autonomous MRD agent with Gemini integration."""
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Register handlers:
    # - Planning and Synthesis use Gemini-powered handlers
    # - Research and Validation use mocked versions
    orchestrator.register_handler(AgentState.PLANNING, inject_services(planning_handler))
    orchestrator.register_handler(AgentState.RESEARCH, research_handler)
    orchestrator.register_handler(AgentState.SYNTHESIS, inject_services(synthesis_handler))
    orchestrator.register_handler(AgentState.VALIDATION, validation_handler)
    
    # User prompt
    user_prompt = """
    I want to build a skill-based gambling app targeting young men, 
    similar to Triumph but for the European market.  
    Help me understand the competitive landscape, regulatory requirements,
    and identify opportunities in the IO gaming space.
    """
    
    # Run the pipeline
    print("=" * 60)
    print("AUTONOMOUS MRD AGENT WITH GEMINI")
    print("=" * 60)
    
    result = await orchestrator.run(user_prompt)
    
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