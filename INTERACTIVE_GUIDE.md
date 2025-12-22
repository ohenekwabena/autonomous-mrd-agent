# 🤖 Interactive MRD Agent - User Guide

## Overview
The Autonomous MRD Agent accepts your product idea and generates a comprehensive Market Requirements Document with research, analysis, and strategic recommendations.

## Quick Start

### 1. Run the Interactive Agent
```bash
python main.py
```

### 2. Enter Your Product Idea
When prompted, describe your product concept. Examples:

```
✅ Good prompts:
- "Build a skill-based gambling app for Europe, similar to Triumph"
- "Create a mobile fitness app targeting Gen Z users with AI coaching"
- "Develop a B2B SaaS platform for restaurant inventory management"
- "Build a social travel planning app for millennials"

❌ Too vague:
- "Make an app"
- "Something with AI"
```

### 3. Watch the Pipeline Execute

The agent progresses through 5 stages:

```
📋 PLANNING (5-10 seconds)
   ↓ Gemini interprets your prompt and creates research tasks
   
🔬 RESEARCH (20-40 seconds)
   ↓ Generates competitor data, market analysis, regulations, sentiment
   ↓ Real-time progress: ✓ completed, ❌ failed
   
🧬 SYNTHESIS (10-20 seconds)
   ↓ Gemini combines research into structured MRD
   
✅ VALIDATION (2-5 seconds)
   ↓ Checks for quality, completeness, citation requirements
   
💾 COMPLETE
   ↓ Saves JSON + Markdown files to output/ folder
```

### 4. Review Your Outputs

After completion, check the `output/` directory:

```
output/
├── mrd_<session>_<timestamp>.json    # Full structured data
└── mrd_<session>_<timestamp>.md      # Human-readable summary
```

## Output Contents

### JSON File Contains:
- Executive summary
- Market overview with citations
- Competitor profiles (5-10 apps)
- SWOT analysis
- Target audience segments
- Regulatory analysis by region
- Feature recommendations with priority/effort
- Go-to-market strategy
- Risk analysis and mitigations

### Markdown File Shows:
- Executive summary
- Key competitors
- SWOT summary
- Top feature recommendations
- Regulatory status
- Easy-to-read formatting

## Progress Display

### Real-Time Research Updates
```
🔬 Research Phase - Executing 8 tasks...
  ✓ competitor_002: 4 competitor(s) analyzed
  ✓ sentiment_004: tiktok sentiment analyzed (sample: 3250)
  ✓ regulatory_005: 2 region(s) checked - UK, EU
  ✓ market_001: market data collected
  ❌ gap_007: API rate limit exceeded
```

### Data Collection Summary
```
📦 Data Collected:
  • Competitors: 9
  • Market Data: Yes
  • Regulatory Statuses: 4
  • Sentiment Analyses: 2
  • Gap Opportunities: 3
  • Overall Completeness: 86.7%
```

### Synthesis Output
```
🧬 Synthesis Phase - Generating MRD...
  ✅ MRD generated successfully (ID: mrd-6b6d91a1)
     • Competitors: 5
     • Features: 3
     • Regulatory regions: 2
     • Target audiences: 2
     • Overall confidence: high
```

## Example Output

See `output/demo_mrd_sample.md` for a complete example MRD.

## Handling Rate Limits

If you see `429 RESOURCE_EXHAUSTED` errors:

1. **Wait 60 seconds** - Gemini free tier has limits (10 requests/minute)
2. **Reduce concurrent tasks** - The agent auto-retries failed tasks
3. **Use demo output** - See `output/demo_mrd_sample.md` for reference

The agent will:
- ✅ Complete whatever tasks succeed
- ⚠️ Mark failed tasks and retry them
- 🔄 Continue if data completeness ≥35%
- ❌ Error if too many failures

## Tips for Best Results

### 1. Be Specific
```
Good: "Build a meal planning app for busy parents with dietary restrictions"
Bad:  "Make a food app"
```

### 2. Mention Your Target Market
```
Good: "Create a fitness app for Gen Z focusing on social workouts"
Bad:  "Create a fitness app"
```

### 3. Reference Competitors (Optional)
```
Good: "Build a project management tool like Asana but for creative teams"
Bad:  "Build a project management tool"
```

### 4. Specify Region (If Relevant)
```
Good: "Develop a payments platform for Southeast Asian SMBs"
Bad:  "Develop a payments platform"
```

## Architecture

```
User Prompt
    ↓
Planning Agent (Gemini) → 6-8 Research Tasks
    ↓
Research Agents (Gemini) → Competitor, Market, Regulatory, Sentiment Data
    ↓
Synthesis Agent (Gemini) → Structured MRD JSON
    ↓
Validation Agent → Quality Checks
    ↓
Save to output/ → JSON + Markdown
```

## Features

✅ **Interactive CLI** - Enter custom prompts  
✅ **Real-time progress** - See each stage complete  
✅ **Rich display** - Emojis, colors, structured output  
✅ **Dual formats** - JSON for database, Markdown for humans  
✅ **Auto-save** - Files saved to `output/` directory  
✅ **Error handling** - Graceful failures with retries  
✅ **Quality gates** - Validation before completion  
✅ **Citation tracking** - Every claim backed by sources  

## Troubleshooting

### Error: API Key Not Set
```bash
# Windows
$env:GEMINI_API_KEY = "your-api-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

### Error: Rate Limit Exceeded
- Wait 60 seconds before running again
- Gemini free tier: 10 requests/minute
- Agent will retry failed tasks automatically

### Error: No MRD Generated
- Check completeness % - needs ≥35% data
- Some research tasks may have failed
- Review error messages for details

### Low Data Completeness
- Normal for rate-limited runs
- Agent uses fallback mechanisms
- MRD will have lower confidence rating

## What's Next?

After generating your MRD:

1. **Review the output** - Check `output/mrd_*.md` file
2. **Validate assumptions** - Research gaps are listed
3. **Deep-dive competitors** - Use the competitor list as starting point
4. **Check regulations** - Consult legal for your specific case
5. **Prioritize features** - Focus on "must_have" recommendations
6. **Plan MVP** - Use feature recommendations to scope v1.0

## Support

For issues or questions:
- Check `GENERAL_BRIEF.md` for architecture details
- Review error messages for specific issues
- Ensure API key is set correctly
- Wait 60s between runs to avoid rate limits

---

**Built with:** Python 3.11+ • Gemini 2.0 Flash • Pydantic v2 • asyncio
