"""
Planning handlers powered by Gemini.
Interpret user prompt and generate structured research plan.
"""

import json
import uuid
from typing import Optional
from models.core import ResearchPlan, ResearchTask
from orchestration.state_machine import OrchestratorContext


class PromptInterpreter:
    """Interpret user prompt using Gemini with fallback to mock."""
    
    def __init__(self, llm):
        self.llm = llm

    async def interpret(self, user_prompt: str) -> dict:
        """Extract goal, constraints, entities, regions from user prompt."""
        system = (
            "You are a product strategy analyst. Extract the following from a user prompt:\n"
            "- goal: what the user wants to build or analyze\n"
            "- constraints: any limitations or requirements\n"
            "- entities: specific products, companies, or apps mentioned\n"
            "- regions: geographic regions of interest\n"
            "Return as JSON with these keys."
        )
        prompt = f"User prompt: {user_prompt}"
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.3)
        except Exception as e:
            print(f"[PLANNING ERROR] Gemini failed: {type(e).__name__}: {str(e)}")
            print(f"[PLANNING WARNING] Using mock interpretation")
            # Mock fallback
            result = {
                "goal": "Build a skill-based gambling application",
                "constraints": ["regulatory compliance", "responsible gambling features"],
                "entities": ["gambling platforms", "skill games"],
                "regions": ["global"]
            }
        return result


class PlanGenerator:
    """Generate research tasks from interpreted goal using Gemini with fallback."""
    
    def __init__(self, llm):
        self.llm = llm

    async def generate_plan(self, user_prompt: str, interpreted: dict) -> ResearchPlan:
        """Create a structured research plan with tasks."""
        system = (
            "You are a research planning expert. Generate a detailed research plan with tasks.\n"
            "Each task must have:\n"
            "- task_id: unique identifier (string)\n"
            "- task_type: one of 'market', 'competitor', 'sentiment', 'regulatory', 'gap_analysis'\n"
            "- query: the research question (min 10 chars)\n"
            "- target_entities: list of items to research (companies, apps, regions)\n"
            "- required_tools: list of tool names needed\n"
            "- success_criteria: what makes this task complete\n"
            "- max_retries: 1-5\n"
            "- timeout_seconds: 30-120\n"
            "\n"
            "Return JSON with:\n"
            "- plan_id: unique plan identifier\n"
            "- interpreted_goal: your understanding of the goal\n"
            "- tasks: list of ResearchTask objects\n"
            "- estimated_duration_minutes: estimate\n"
        )
        
        prompt = (
            f"User prompt: {user_prompt}\n\n"
            f"Interpreted analysis:\n{json.dumps(interpreted, indent=2)}\n\n"
            "Generate a comprehensive research plan."
        )
        
        try:
            result = await self.llm.generate_json(system, prompt, temperature=0.2)
        except Exception as e:
            print(f"[PLANNING ERROR] Gemini failed: {type(e).__name__}: {str(e)}")
            print(f"[PLANNING WARNING] Using mock plan")
            result = {
                "plan_id": f"plan-{uuid.uuid4().hex[:8]}",
                "interpreted_goal": interpreted.get("goal", "Build skill-based gambling app"),
                "tasks": [
                    {
                        "task_id": "task-market-1",
                        "task_type": "market",
                        "query": "What is the current market size for skill-based gaming?",
                        "target_entities": ["skill gaming industry", "online gaming platforms"],
                        "required_tools": ["market_data", "industry_reports"],
                        "success_criteria": "Gathered market sizing and growth trends",
                        "max_retries": 3,
                        "timeout_seconds": 60
                    },
                    {
                        "task_id": "task-competitor-1",
                        "task_type": "competitor",
                        "query": "Who are the main competitors in skill-based gambling?",
                        "target_entities": ["DraftKings", "FanDuel", "Skillz", "Pogo"],
                        "required_tools": ["competitor_analysis", "web_search"],
                        "success_criteria": "Identified top 5 competitors and their features",
                        "max_retries": 3,
                        "timeout_seconds": 60
                    },
                    {
                        "task_id": "task-regulatory-1",
                        "task_type": "regulatory",
                        "query": "What are the regulatory requirements for skill-based gambling by region?",
                        "target_entities": ["US", "EU", "APAC", "Canada"],
                        "required_tools": ["regulatory_database", "legal_research"],
                        "success_criteria": "Documented compliance requirements per region",
                        "max_retries": 3,
                        "timeout_seconds": 90
                    }
                ],
                "estimated_duration_minutes": 30
            }
        
        # Parse and validate tasks
        tasks = []
        for task_data in result.get("tasks", []):
            try:
                task = ResearchTask(
                    task_id=task_data.get("task_id", f"task-{uuid.uuid4().hex[:8]}"),
                    task_type=task_data.get("task_type", "market"),
                    query=task_data.get("query", ""),
                    target_entities=task_data.get("target_entities", []),
                    required_tools=task_data.get("required_tools", []),
                    success_criteria=task_data.get("success_criteria", ""),
                    max_retries=task_data.get("max_retries", 3),
                    timeout_seconds=task_data.get("timeout_seconds", 30)
                )
                tasks.append(task)
            except Exception as e:
                print(f"[WARN] Failed to parse task: {e}")
        
        return ResearchPlan(
            plan_id=result.get("plan_id", f"plan-{uuid.uuid4().hex[:8]}"),
            user_intent=user_prompt,
            interpreted_goal=result.get("interpreted_goal", ""),
            tasks=tasks if tasks else [
                # Fallback: minimal plan
                ResearchTask(
                    task_id="fallback-competitor",
                    task_type="competitor",
                    query="Analyze key competitors in the market",
                    target_entities=[],
                    required_tools=["search_sensor_tower"],
                    success_criteria="Identify top 3 competitors"
                )
            ],
            estimated_duration_minutes=result.get("estimated_duration_minutes", 30)
        )


async def planning_handler(ctx: OrchestratorContext) -> OrchestratorContext:
    """
    Planning handler powered by Gemini.
    Interprets user prompt and generates a research plan.
    """
    print(f"[PLANNING] Interpreting prompt with Gemini: {ctx.user_prompt[:50]}...")
    
    try:
        interpreter = PromptInterpreter(ctx.services.llm)
        interpreted = await interpreter.interpret(ctx.user_prompt)
        print(f"[PLANNING] Interpreted goal: {interpreted.get('goal', 'N/A')}")
        
        planner = PlanGenerator(ctx.services.llm)
        ctx.research_plan = await planner.generate_plan(ctx.user_prompt, interpreted)
        
        print(f"[PLANNING] Generated plan with {len(ctx.research_plan.tasks)} tasks")
        for task in ctx.research_plan.tasks:
            print(f"  - {task.task_id}: {task.task_type} ({', '.join(task.required_tools)})")
        
    except Exception as e:
        print(f"[PLANNING ERROR] {e}")
        ctx.errors.append(f"Planning failed: {str(e)}")
        raise
    
    return ctx
