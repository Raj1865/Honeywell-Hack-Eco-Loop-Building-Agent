"""
LLM Client Wrapper
===================
Thin wrapper around Ollama (or compatible) API for tool-calling conversations.
Handles structured tool definitions, streaming, retries, and multi-format tool extraction.
"""

import json
import re
import time
from typing import Optional

import httpx
from loguru import logger


class LLMClient:
    """
    Client for interacting with a locally-hosted LLM (via Ollama API).
    
    Supports:
    - Chat completions with tool definitions
    - Multi-format tool call parsing (Native API tool_calls + Markdown JSON code blocks)
    - Automatic retries with exponential backoff
    - Structured JSON output parsing
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout_seconds)

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Send a chat completion request with optional tool definitions.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if tools:
            payload["tools"] = tools

        return self._request_with_retry(payload)

    def _request_with_retry(self, payload: dict) -> dict:
        """Execute request with exponential backoff retry."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                response = self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"LLM response received in {elapsed:.2f}s "
                        f"(tokens: {result.get('eval_count', '?')})"
                    )
                    return result
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    logger.warning(f"LLM request failed (attempt {attempt}): {last_error}")

            except httpx.TimeoutException:
                last_error = f"Timeout after {self.timeout_seconds}s"
                logger.warning(f"LLM request timed out (attempt {attempt})")

            except httpx.ConnectError:
                last_error = f"Cannot connect to {self.base_url}"
                logger.error(f"LLM connection failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"LLM request error (attempt {attempt}): {e}")

            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

        raise ConnectionError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    def extract_tool_calls(self, response: dict) -> list[dict]:
        """
        Extract tool calls from an LLM response.
        Supports native Ollama function calling and text content JSON parsing.
        """
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        parsed = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name:
                parsed.append({"name": name, "arguments": args})

        if not parsed:
            content = message.get("content", "")
            parsed.extend(self._parse_tool_calls_from_text(content))

        return parsed

    def _parse_tool_calls_from_text(self, text: str) -> list[dict]:
        """Parse tool calls embedded in JSON text content."""
        if not text:
            return []

        parsed = []
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        candidates = json_blocks if json_blocks else [text]

        for block in candidates:
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict):
                    actions = data.get("actions", [])
                    if isinstance(actions, list):
                        for item in actions:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get("tool", "")
                                args = item.get("arguments") or item.get("args", {})
                                if name:
                                    parsed.append({"name": name, "arguments": args})
                    elif "name" in data and ("arguments" in data or "args" in data):
                        name = data.get("name", "")
                        args = data.get("arguments") or data.get("args", {})
                        if name:
                            parsed.append({"name": name, "arguments": args})
            except Exception:
                continue

        return parsed

    def extract_content(self, response: dict) -> str:
        """Extract text content from LLM response."""
        return response.get("message", {}).get("content", "")

    def health_check(self) -> bool:
        """Check if LLM server is reachable and model loaded."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(self.model in name for name in model_names):
                    logger.info(f"LLM health check passed. Model '{self.model}' available.")
                    return True
                else:
                    logger.warning(
                        f"Model '{self.model}' not found. Available: {model_names}"
                    )
                    return False
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False

    def close(self):
        """Close HTTP client."""
        self._client.close()
