"""
Use Gemini to generate realistic research data instead of mocking.
This gives us actual AI-powered research while maintaining the same interface.
"""

import json
import asyncio
from adapters.gemini_client import GeminiClient


class ResearchGenerator:
    """Use Gemini to generate realistic research data for various research types"""
    
    def __init__(self, llm: GeminiClient):
        self.llm = llm
    
    async def generate_competitor_research(self, app_names: list[str]) -> dict:
        """Generate realistic competitor analysis using Gemini"""
        prompt = f"""
        Generate realistic competitor analysis data for these apps: {', '.join(app_names)}
        
        Return JSON with this structure for EACH app:
        {{
            "app_name": "string",
            "monthly_downloads": integer,
            "revenue_estimate": "string like $1.5M/month",
            "top_countries": ["country_codes"],
            "category_rank": integer,
            "user_retention_30d": float between 0 and 1,
            "key_features": ["feature1", "feature2"],
            "marketing_channels": ["channel1", "channel2"]
        }}
        
        Return a JSON array of these objects. Use realistic estimates based on actual market knowledge.
        """
        
        system = (
            "You are a market research analyst. Generate realistic competitor data based on "
            "actual market knowledge and app store trends. Return ONLY valid JSON array."
        )
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.7)
            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def generate_regulatory_research(self, regions: list[str]) -> dict:
        """Generate realistic regulatory analysis using Gemini"""
        prompt = f"""
        Generate realistic regulatory compliance data for skill-based gaming apps in these regions: {', '.join(regions)}
        
        For EACH region, return JSON with this structure:
        {{
            "region": "country/region code",
            "status": "legal|illegal|gray_area|requires_license",
            "authority": "regulatory authority name",
            "license_types": ["license_type1"],
            "restrictions": ["restriction1", "restriction2"],
            "recent_changes": "description of recent regulatory changes",
            "skill_game_distinction": "whether skill games are treated differently"
        }}
        
        Return a JSON array of these objects. Use accurate regulatory information.
        """
        
        system = (
            "You are a regulatory compliance expert. Generate accurate regulatory data for "
            "skill-based gaming in different regions. Return ONLY valid JSON array."
        )
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.5)
            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def generate_sentiment_analysis(self, platform: str, query: str) -> dict:
        """Generate realistic sentiment analysis using Gemini"""
        prompt = f"""
        Generate realistic social sentiment analysis for {platform} with query: "{query}"
        
        Return JSON with this EXACT structure:
        {{
            "platform": "{platform}",
            "sample_size": integer between 500 and 5000,
            "sentiment": {{
                "positive": float between 0 and 1,
                "negative": float between 0 and 1,
                "neutral": float between 0 and 1
            }},
            "top_hashtags": ["hashtag1", "hashtag2", "hashtag3"],
            "top_themes": {{
                "positive": ["theme1", "theme2", "theme3"],
                "negative": ["theme1", "theme2"]
            }},
            "influencer_mentions": [
                {{"handle": "@creator1", "followers": 100000}},
                {{"handle": "@creator2", "followers": 50000}}
            ],
            "engagement_rate": 0.035,
            "trend_direction": "rising"
        }}
        
        CRITICAL: sentiment.positive + sentiment.negative + sentiment.neutral MUST equal 1.0
        Use realistic numbers based on typical social media engagement for gaming apps.
        Return ONLY valid JSON.
        """
        
        system = (
            "You are a social media analyst. Generate realistic sentiment data based on "
            "typical social media trends for gaming apps. Return ONLY valid JSON with exact structure."
        )
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.6)
            # Normalize sentiment ratios to ensure they sum to 1.0
            if isinstance(result, dict) and "sentiment" in result:
                sent = result["sentiment"]
                if isinstance(sent, dict):
                    pos = float(sent.get("positive", 0.4))
                    neg = float(sent.get("negative", 0.2))
                    neu = float(sent.get("neutral", 0.4))
                    total = pos + neg + neu
                    if total > 0:
                        sent["positive"] = round(pos / total, 3)
                        sent["negative"] = round(neg / total, 3)
                        sent["neutral"] = round(neu / total, 3)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def generate_market_research(self, market_query: str) -> dict:
        """Generate realistic market research using Gemini"""
        prompt = f"""
        Generate realistic market research data for: {market_query}
        
        Return JSON with this structure:
        {{
            "market_size_usd": integer,
            "growth_rate_percent": float,
            "market_segments": ["segment1", "segment2"],
            "key_trends": ["trend1", "trend2", "trend3"],
            "barriers_to_entry": ["barrier1", "barrier2"],
            "success_factors": ["factor1", "factor2"],
            "competitive_landscape": "description",
            "forecast_period_years": integer,
            "cagr_percent": float
        }}
        
        Use realistic market data and trends. Return ONLY valid JSON.
        """
        
        system = (
            "You are a market research analyst. Generate realistic market research data "
            "based on actual industry trends and data. Return ONLY valid JSON."
        )
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.6)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
