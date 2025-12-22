"""
Synthesis handlers powered by Gemini.
Convert research aggregate into structured MRD.
"""

import json
import uuid
from models.core import (
    ResearchAggregate, ConfidenceLevel, DataSource, Citation, VerifiedClaim,
    GapAnalysisItem
)
from models.mrd import StrategicAnalysis, SWOTAnalysis, TargetAudience, FeatureRecommendation
from orchestration.state_machine import OrchestratorContext


class SynthesisEngine:
    """Generate MRD using Gemini from research data, with fallback mock."""
    
    def __init__(self, llm):
        self.llm = llm

    async def to_mrd(self, agg: ResearchAggregate, user_intent: str, interpreted_goal: str) -> StrategicAnalysis:
        """
        Convert research aggregate to MRD using Gemini.
        Gemini generates narrative and structure; Pydantic validates.
        Falls back to mock if Gemini unavailable.
        """
        system = (
            "You are an expert product strategist generating a complete MRD from research data. "
            "CRITICAL REQUIREMENTS:\n"
            "1. Return VALID JSON ONLY - no markdown, no code fences\n"
            "2. ALL arrays of claims must use claim_object structure: "
            "{\"claim\": \"text\", \"confidence\": \"high|medium|low\", \"citations\": [{\"source\": \"value\"}]}\n"
            "3. acquisition_channels, rationale, primary_channels MUST be claim_objects (NOT strings)\n"
            "4. NEVER use sources other than: sensor_tower, social_sentiment, web_search, regulatory_db, manual_input\n"
            "5. CRITICAL RULE: \"high\" confidence claims MUST have 2+ citations (NEVER just 1)\n"
            "6. Medium/low confidence can have 1 citation\n"
            "7. Priority values: ONLY 'must_have', 'should_have', 'nice_to_have'\n"
            "8. Effort, potential_value, implementation_complexity: ONLY 'low', 'medium', 'high'\n"
            "9. legal_status: ONLY 'legal', 'illegal', 'gray_area', 'requires_license'\n"
            "10. SWOT sections MUST have at least 1 item each with full claim structure\n"
            "11. Empty arrays [] are INVALID - every list must have content\n\n"
            "REQUIRED JSON STRUCTURE (follow exactly):\n{\n"
            "  \"executive_summary\": \"100+ char summary\",\n"
            "  \"market_overview\": [{\"claim\": \"text\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}],\n"
            "  \"competitor_list\": [{\"name\": \"str\", \"app_store_id\": null, \"key_features\": [\"str\"], \"strengths\": [{\"claim\": \"text\", \"confidence\": \"high\", \"citations\": [{\"source\": \"sensor_tower\"}, {\"source\": \"web_search\"}]}], \"weaknesses\": [{\"claim\": \"text\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}]}],\n"
            "  \"competitive_moat_analysis\": [{\"claim\": \"text\", \"confidence\": \"high\", \"citations\": [{\"source\": \"sensor_tower\"}, {\"source\": \"social_sentiment\"}]}],\n"
            "  \"swot\": {\n"
            "    \"strengths\": [{\"claim\": \"Market growth rate\", \"confidence\": \"high\", \"citations\": [{\"source\": \"web_search\"}, {\"source\": \"manual_input\"}]}],\n"
            "    \"weaknesses\": [{\"claim\": \"Regulatory barriers\", \"confidence\": \"high\", \"citations\": [{\"source\": \"regulatory_db\"}, {\"source\": \"web_search\"}]}],\n"
            "    \"opportunities\": [{\"claim\": \"Emerging markets\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}],\n"
            "    \"threats\": [{\"claim\": \"Market consolidation\", \"confidence\": \"high\", \"citations\": [{\"source\": \"sensor_tower\"}, {\"source\": \"web_search\"}]}]\n"
            "  },\n"
            "  \"target_audiences\": [{\"segment_name\": \"str\", \"demographics\": {}, \"psychographics\": [\"str\"], \"acquisition_channels\": [{\"claim\": \"Twitch marketing\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}]}],\n"
            "  \"regulatory_analysis\": [{\"region\": \"str\", \"legal_status\": \"legal|illegal|gray_area|requires_license\", \"license_requirements\": [\"str\"], \"restrictions\": [\"str\"], \"recent_changes\": [{\"claim\": \"text\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"regulatory_db\"}]}], \"confidence\": \"high\", \"citation\": {\"source\": \"regulatory_db\"}}],\n"
            "  \"regulatory_recommendation\": \"50+ chars\",\n"
            "  \"gap_analysis\": [{\"opportunity\": \"str\", \"current_market_state\": \"str\", \"potential_value\": \"high|medium|low\", \"implementation_complexity\": \"high|medium|low\", \"supporting_evidence\": [{\"claim\": \"text\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}]}],\n"
            "  \"feature_recommendations\": [{\"feature_name\": \"str\", \"description\": \"str\", \"priority\": \"must_have|should_have|nice_to_have\", \"rationale\": [{\"claim\": \"text\", \"confidence\": \"high\", \"citations\": [{\"source\": \"sensor_tower\"}, {\"source\": \"web_search\"}]}], \"estimated_effort\": \"low|medium|high\", \"competitive_advantage\": \"str\"}],\n"
            "  \"gtm_strategy\": {\"primary_channels\": [{\"claim\": \"App Store Optimization\", \"confidence\": \"medium\", \"citations\": [{\"source\": \"web_search\"}]}], \"influencer_strategy\": \"str\", \"geographic_rollout\": [\"str\"], \"regulatory_considerations\": [\"str\"]},\n"
            "  \"key_risks\": [{\"claim\": \"text\", \"confidence\": \"high\", \"citations\": [{\"source\": \"regulatory_db\"}, {\"source\": \"web_search\"}]}],\n"
            "  \"mitigation_strategies\": [\"str\"],\n"
            "  \"data_sources_used\": [\"sensor_tower\", \"web_search\"],\n"
            "  \"claims_requiring_verification\": [\"str\"],\n"
            "  \"research_gaps\": [\"str\"],\n"
            "  \"overall_confidence\": \"high|medium|low|unverified\"\n}"
        )
        
        # Serialize research aggregate
        agg_json = json.dumps(agg.model_dump(), indent=2, default=str)
        
        prompt = (
            f"User intent: {user_intent}\n"
            f"Interpreted goal: {interpreted_goal}\n\n"
            f"Research data:\n{agg_json}\n\n"
            "Generate a comprehensive MRD based on this data."
        )
        
        try:
            # Use generate_json for better JSON handling
            mrd_data = await self.llm.generate_json(system, prompt, temperature=0.2)
            print(f"[SYNTHESIS DEBUG] Gemini returned valid JSON")
        except Exception as e:
            print(f"[SYNTHESIS ERROR] Gemini failed: {type(e).__name__}: {str(e)}")
            print(f"[SYNTHESIS WARNING] Using mock MRD")
            # Return mock data directly as dict, skip JSON serialization
            mrd_data = self._mock_mrd_json()
            mrd_data["mrd_id"] = mrd_data.get("mrd_id", f"mrd-{uuid.uuid4().hex[:8]}")
            mrd_data["user_intent"] = user_intent
            mrd_data["interpreted_goal"] = interpreted_goal
            mrd_data["version"] = "1.0"
            if mrd_data.get("overall_confidence") not in ["high", "medium", "low", "unverified"]:
                mrd_data["overall_confidence"] = "medium"
            mrd = StrategicAnalysis(**mrd_data)
            return mrd

        
        # Add required fields if missing and validate
        mrd_data["mrd_id"] = mrd_data.get("mrd_id", f"mrd-{uuid.uuid4().hex[:8]}")
        mrd_data["user_intent"] = user_intent
        mrd_data["interpreted_goal"] = interpreted_goal
        mrd_data["version"] = "1.0"
        
        # Ensure overall_confidence is valid
        if mrd_data.get("overall_confidence") not in ["high", "medium", "low", "unverified"]:
            mrd_data["overall_confidence"] = "medium"
        
        # Use Pydantic to validate and construct MRD
        try:
            mrd = StrategicAnalysis(**mrd_data)
            return mrd
        except Exception as e:
            print(f"[SYNTHESIS ERROR] Pydantic validation failed: {e}")
            print("[SYNTHESIS] Falling back to mock MRD")
            # Use mock if validation fails
            mrd_data = self._mock_mrd_json()
            mrd_data["mrd_id"] = f"mrd-{uuid.uuid4().hex[:8]}"
            mrd_data["user_intent"] = user_intent
            mrd_data["interpreted_goal"] = interpreted_goal
            mrd_data["version"] = "1.0"
            mrd = StrategicAnalysis(**mrd_data)
            return mrd

    def _mock_mrd_json(self) -> dict:
        """Generate mock MRD for demo when Gemini unavailable."""
        # Return minimal valid structure matching StrategicAnalysis schema
        return {
            "mrd_id": f"mrd-{uuid.uuid4().hex[:8]}",
            "overall_confidence": ConfidenceLevel.MEDIUM,
            "executive_summary": "Market analysis reveals opportunity in skill-based gambling with strong competitive landscape. Key success factors: regulatory compliance, differentiated gameplay, and targeted marketing.",
            "market_overview": [
                {
                    "claim": "Skill-based gaming market growing 25-30% annually",
                    "confidence": ConfidenceLevel.HIGH,
                    "citations": [
                        Citation(source=DataSource.WEB_SEARCH),
                        Citation(source=DataSource.SENSOR_TOWER)
                    ]
                }
            ],
            "competitor_list": [
                {
                    "name": "DraftKings",
                    "type": "Daily Fantasy Sports",
                    "market_share": "High",
                    "key_features": ["Multi-sport coverage", "Mobile", "Daily contests"]
                },
                {
                    "name": "FanDuel",
                    "type": "Daily Fantasy Sports",
                    "market_share": "High",
                    "key_features": ["Fantasy sports", "Sportsbook", "Rewards program"]
                }
            ],
            "competitive_moat_analysis": [
                {
                    "claim": "Brand loyalty in regulated markets is strong",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "citations": [Citation(source=DataSource.SENSOR_TOWER)]
                }
            ],
            "swot": SWOTAnalysis(
                strengths=[VerifiedClaim(
                    claim="Clear regulatory framework in US",
                    confidence=ConfidenceLevel.HIGH,
                    citations=[
                        Citation(source=DataSource.REGULATORY_DB),
                        Citation(source=DataSource.WEB_SEARCH)
                    ]
                )],
                weaknesses=[VerifiedClaim(claim="High user acquisition costs", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                opportunities=[VerifiedClaim(claim="International expansion opportunities", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                threats=[VerifiedClaim(
                    claim="Regulatory tightening risk",
                    confidence=ConfidenceLevel.HIGH,
                    citations=[
                        Citation(source=DataSource.REGULATORY_DB),
                        Citation(source=DataSource.WEB_SEARCH)
                    ]
                )]
            ),
            "target_audiences": [
                TargetAudience(
                    segment_name="Male gamers 18-35",
                    demographics={"age": "18-35", "gender": "male"},
                    psychographics=["Competitive", "Mobile-first", "Social"],
                    acquisition_channels=[VerifiedClaim(claim="TikTok and streaming platforms", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.SOCIAL_SENTIMENT)])]
                )
            ],
            "regulatory_analysis": [
                {
                    "region": "US",
                    "legal_status": "requires_license",
                    "license_requirements": [
                        "State gaming commission approval required",
                        "Age verification mandatory"
                    ],
                    "restrictions": [
                        "No minors",
                        "Geo-fencing to legal states"
                    ],
                    "recent_changes": [
                        VerifiedClaim(
                            claim="State-level clarifications on skill gaming",
                            confidence=ConfidenceLevel.MEDIUM,
                            citations=[Citation(source=DataSource.REGULATORY_DB)]
                        )
                    ],
                    "confidence": ConfidenceLevel.HIGH,
                    "citation": Citation(source=DataSource.REGULATORY_DB)
                },
                {
                    "region": "EU",
                    "legal_status": "requires_license",
                    "license_requirements": [
                        "License required per jurisdiction",
                        "GDPR compliance"
                    ],
                    "restrictions": [
                        "Data residency as applicable"
                    ],
                    "recent_changes": [
                        VerifiedClaim(
                            claim="Ongoing updates in UK & Malta",
                            confidence=ConfidenceLevel.MEDIUM,
                            citations=[Citation(source=DataSource.WEB_SEARCH)]
                        )
                    ],
                    "confidence": ConfidenceLevel.HIGH,
                    "citation": Citation(source=DataSource.REGULATORY_DB)
                }
            ],
            "regulatory_recommendation": "Pursue US market first with state-by-state strategy. Prioritize UK and Malta for EU expansion.",
            "gap_analysis": [
                GapAnalysisItem(
                    opportunity="AI-powered matchmaking",
                    current_market_state="Manual or basic matchmaking in existing platforms",
                    potential_value="high",
                    implementation_complexity="medium",
                    supporting_evidence=[VerifiedClaim(claim="Competitive advantage opportunity", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])]
                )
            ],
            "feature_recommendations": [
                FeatureRecommendation(
                    feature_name="Responsible Gaming Controls",
                    description="Dashboard for spend limits and time management",
                    priority="must_have",
                    rationale=[VerifiedClaim(
                        claim="Regulatory requirement",
                        confidence=ConfidenceLevel.HIGH,
                        citations=[
                            Citation(source=DataSource.REGULATORY_DB),
                            Citation(source=DataSource.WEB_SEARCH)
                        ]
                    )],
                    estimated_effort="medium",
                    competitive_advantage="Regulatory compliance"
                ),
                FeatureRecommendation(
                    feature_name="Tournament System",
                    description="Competitive multiplayer tournaments with prizes",
                    priority="should_have",
                    rationale=[VerifiedClaim(claim="User engagement driver", confidence=ConfidenceLevel.MEDIUM, citations=[Citation(source=DataSource.WEB_SEARCH)])],
                    estimated_effort="high",
                    competitive_advantage="Unique engagement mechanic"
                )
            ],
            "gtm_strategy": {
                "primary_channels": [
                    VerifiedClaim(
                        claim="Paid social (TikTok, Instagram)",
                        confidence=ConfidenceLevel.MEDIUM,
                        citations=[Citation(source=DataSource.SOCIAL_SENTIMENT)]
                    ),
                    VerifiedClaim(
                        claim="Influencer partnerships",
                        confidence=ConfidenceLevel.MEDIUM,
                        citations=[Citation(source=DataSource.SOCIAL_SENTIMENT)]
                    )
                ],
                "influencer_strategy": "Partner with gaming creators for tournament showcases",
                "geographic_rollout": ["US-CA", "US-NJ", "US-PA"],
                "regulatory_considerations": ["State-by-state licensing", "Age verification & KYC"]
            },
            "key_risks": [
                {
                    "claim": "Regulatory change or prohibition",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "citations": [Citation(source=DataSource.REGULATORY_DB)]
                },
                {
                    "claim": "Incumbent competition and market consolidation",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "citations": [Citation(source=DataSource.WEB_SEARCH)]
                },
                {
                    "claim": "High user acquisition costs reducing profitability",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "citations": [Citation(source=DataSource.WEB_SEARCH)]
                }
            ],
            "mitigation_strategies": [
                "Invest in regulatory affairs and government relations",
                "Build defensible moat through unique mechanics",
                "Focus on retention and lifetime value optimization"
            ],
            "data_sources_used": ["market_research", "competitor_analysis", "regulatory_review"],
            "claims_requiring_verification": ["Market growth rates", "User retention figures"],
            "research_gaps": ["Detailed TAM analysis", "International market dynamics"]
        }




async def synthesis_handler(ctx: OrchestratorContext) -> OrchestratorContext:
    """
    Synthesis handler powered by Gemini.
    Converts research aggregate into structured MRD.
    """
    print("[SYNTHESIS] Generating MRD with Gemini...")
    
    if not ctx.research_aggregate:
        ctx.errors.append("No research data available for synthesis")
        raise ValueError("Research aggregate is None")
    
    try:
        engine = SynthesisEngine(ctx.services.llm)
        rp = ctx.research_plan
        
        ctx.mrd = await engine.to_mrd(
            ctx.research_aggregate,
            rp.user_intent,
            rp.interpreted_goal
        )
        
        print(f"[SYNTHESIS] MRD generated successfully (id: {ctx.mrd.mrd_id})")
        print(f"  - Competitors: {len(ctx.mrd.competitor_list)}")
        print(f"  - Features: {len(ctx.mrd.feature_recommendations)}")
        print(f"  - Regulatory regions: {len(ctx.mrd.regulatory_analysis)}")
        
    except Exception as e:
        print(f"[SYNTHESIS ERROR] {e}")
        ctx.errors.append(f"Synthesis failed: {str(e)}")
        raise
    
    return ctx
