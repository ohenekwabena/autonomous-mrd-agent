"""
Mock implementations of external tools.
In production, these would be API calls to Sensor Tower, Twitter, regulatory DBs, etc.
Replace these with real API clients while maintaining the ToolResponse interface.
"""

import asyncio
from pydantic import BaseModel
from typing import Optional
import hashlib


class ToolResponse(BaseModel):
    """Standardized response from any tool"""
    success: bool
    data: Optional[dict] = None
    error_message: Optional[str] = None
    raw_response: Optional[str] = None
    tool_name: str = ""
    
    def get_hash(self) -> str:
        """Generate hash of raw response for audit trail"""
        if self.raw_response:
            return hashlib.sha256(self.raw_response.encode()).hexdigest()[:16]
        return ""


class FallbackLadder:
    """
    Defines fallback strategy for tools.
    If primary tool fails, try these alternatives in order.
    
    Strategy: All specific tools fall back to generic web_search
    - search_sensor_tower → web_search
    - analyze_sentiment → web_search  
    - check_regulatory_compliance → web_search
    - web_search (terminal fallback - no alternatives)
    """
    FALLBACK_STRATEGIES = {
        "search_sensor_tower": ["web_search"],
        "analyze_sentiment": ["web_search"],
        "check_regulatory_compliance": ["web_search"],
        "web_search": [],  # No fallback - last resort
    }
    
    @staticmethod
    def get_fallback_tools(primary_tool: str) -> list[str]:
        """Get ordered list of fallback tools for a primary tool"""
        return FallbackLadder.FALLBACK_STRATEGIES.get(primary_tool, [])


class MockToolkit:
    """
    Mock implementations of external tools.
    In production, these would be API calls.
    
    Fallback Strategy:
    - search_sensor_tower → web_search
    - analyze_sentiment → web_search
    - check_regulatory_compliance → web_search
    - web_search (terminal fallback)
    """
    
    @staticmethod
    async def search_sensor_tower(app_name: str) -> ToolResponse:
        """Mock Sensor Tower API for app store analytics"""
        await asyncio.sleep(0.1)  # Simulate API latency
        
        app_normalized = app_name.lower().strip() if isinstance(app_name, str) else str(app_name).lower()
        
        # Simulate different responses based on app
        if "triumph" in app_normalized:
            return ToolResponse(
                success=True,
                data={
                    "app_name": "Triumph",
                    "monthly_downloads": 850000,
                    "revenue_estimate": "$2.5M/month",
                    "top_countries": ["US", "UK", "CA"],
                    "category_rank": 12,
                    "user_retention_30d": 0.35
                },
                raw_response='{"status": "ok", "data": {...}}'
            )
        elif "skillz" in app_normalized:
            return ToolResponse(
                success=True,
                data={
                    "app_name": "Skillz",
                    "monthly_downloads": 320000,
                    "revenue_estimate": "$1.1M/month",
                    "top_countries": ["US"],
                    "category_rank": 45,
                    "user_retention_30d": 0.18
                },
                raw_response='{"status": "ok", "data": {...}}'
            )
        else:
            # Return generic success for other apps
            return ToolResponse(
                success=True,
                data={
                    "app_name": app_name,
                    "monthly_downloads": 500000,
                    "revenue_estimate": "$1.5M/month",
                    "category_rank": 25,
                    "user_retention_30d": 0.25,
                    "note": "Estimated based on category averages"
                },
                raw_response='{"status": "ok", "data": {...}}'
            )
    
    @staticmethod
    async def analyze_sentiment(platform: str, query: str) -> ToolResponse:
        """Mock social sentiment analysis (Twitter/TikTok API)"""
        await asyncio.sleep(0.2)
        
        if platform.lower() == "tiktok":
            return ToolResponse(
                success=True,
                data={
                    "platform": "TikTok",
                    "query": query,
                    "sample_size": 1250,
                    "sentiment": {
                        "positive": 0.62,
                        "negative": 0.15,
                        "neutral": 0.23
                    },
                    "top_hashtags": ["#triumphapp", "#skillgaming", "#winmoney"],
                    "influencer_mentions": [
                        {"handle": "@gaming_mike", "followers": 1200000},
                        {"handle": "@cashmoney_plays", "followers": 890000}
                    ]
                },
                raw_response='{"status": "ok", ...}'
            )
        return ToolResponse(success=False, error_message=f"Platform {platform} not supported")
    
    @staticmethod
    async def check_regulatory_compliance(region: str) -> ToolResponse:
        """Mock regulatory compliance check (regulatory database API)"""
        await asyncio.sleep(0.15)
        
        region_normalized = region.upper() if isinstance(region, str) else str(region).upper()
        
        # More flexible matching - handle variations
        regulations = {
            "UK": {
                "status": "requires_license",
                "authority": "UK Gambling Commission",
                "license_types": ["Remote Gambling License"],
                "restrictions": ["Age verification required", "No credit card deposits"],
                "recent_changes": "2024 white paper tightening rules on advertising"
            },
            "GB": {  # Alternative for UK
                "status": "requires_license",
                "authority": "UK Gambling Commission",
                "license_types": ["Remote Gambling License"],
                "restrictions": ["Age verification required", "No credit card deposits"],
                "recent_changes": "2024 white paper tightening rules on advertising"
            },
            "EU": {
                "status": "gray_area",
                "note": "Each member state has different regulations",
                "generally_legal": ["Malta", "Gibraltar", "Isle of Man"],
                "restricted": ["Germany", "Netherlands", "France"]
            },
            "EUROPE": {
                "status": "gray_area",
                "note": "Each European country has different regulations",
                "generally_legal": ["Malta", "Gibraltar", "Isle of Man"],
                "restricted": ["Germany", "Netherlands", "France"]
            },
            "US": {
                "status": "gray_area",
                "legal_states": ["NJ", "PA", "MI", "WV"],
                "skill_gaming_distinction": "Skill games may be exempt in some states"
            },
            "USA": {
                "status": "gray_area",
                "legal_states": ["NJ", "PA", "MI", "WV"],
                "skill_gaming_distinction": "Skill games may be exempt in some states"
            },
        }
        
        # Try exact match first
        if region_normalized in regulations:
            return ToolResponse(
                success=True,
                data=regulations[region_normalized],
                raw_response=f'{{"region": "{region}", ...}}'
            )
        
        # Try partial match (starts with)
        for key, data in regulations.items():
            if region_normalized.startswith(key) or key.startswith(region_normalized[:2]):
                return ToolResponse(
                    success=True,
                    data=data,
                    raw_response=f'{{"region": "{region}", ...}}'
                )
        
        # Fallback - return generic data
        return ToolResponse(
            success=True,
            data={
                "status": "unknown",
                "region": region,
                "note": "Region not in primary database, may require further research"
            },
            raw_response=f'{{"region": "{region}", ...}}'
        )
    
    @staticmethod
    async def web_search(query: str) -> ToolResponse:
        """Mock web search (Google/Bing API)"""
        await asyncio.sleep(0.1)
        return ToolResponse(
            success=True,
            data={
                "results": [
                    {"title": "Triumph App Review", "url": "https://example.com/1"},
                    {"title": "Skill Gaming Market 2024", "url": "https://example.com/2"}
                ]
            },
            raw_response='{"results": [...]}'
        )
