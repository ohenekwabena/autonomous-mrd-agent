import os
import asyncio
import json
import re
from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key_env: str = "GEMINI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def generate(self, system: str, prompt: str, response_schema=None, temperature: float = 0.2, force_json: bool = False) -> str:
        """Generate content with system and user prompts. Runs in thread pool to handle sync API."""
        def _generate_sync():
            result = self.client.models.generate_content(
                model=self.model,
                contents=f"{system}\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json" if (force_json or response_schema) else "text/plain",
                    response_schema=response_schema
                )
            )
            return result.text
        
        # Run sync call in thread pool
        return await asyncio.to_thread(_generate_sync)

    async def generate_json(self, system: str, prompt: str, temperature: float = 0.2) -> dict:
        """Generate JSON response and parse it, with resilient extraction."""
        text = await self.generate(system, prompt, temperature=temperature, force_json=True)
        # Fast path
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove common code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Remove a leading language hint like 'json' if present
            cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)

        # Try to extract a JSON object or array
        match_obj = re.search(r"\{[\s\S]*\}", cleaned)
        if match_obj:
            candidate = match_obj.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match_arr = re.search(r"\[[\s\S]*\]", cleaned)
        if match_arr:
            candidate = match_arr.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # As a last resort, raise a clear error with a short preview
        preview = cleaned[:200].replace("\n", " ")
        raise json.JSONDecodeError(f"Could not parse JSON from Gemini output: {preview}", cleaned, 0)