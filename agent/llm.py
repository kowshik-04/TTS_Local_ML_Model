"""Streaming chat client for any OpenAI-compatible local or hosted LLM."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class LLMError(RuntimeError):
    pass


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant for Telugu and Telugu-English "
    "(Tenglish) callers. Reply in the language the caller used. Keep spoken "
    "answers to one to three short sentences. Never output markdown, code, URLs, "
    "emoji, bullet lists or stage directions. Ask one brief clarifying question "
    "when a request is ambiguous. Do not claim capabilities the local system "
    "does not have."
)


def parse_sse_line(line: str) -> str | None:
    """Return the text delta from one OpenAI-style SSE line, if present."""
    line = line.strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except ValueError as error:
        raise LLMError(f"LLM stream returned malformed JSON: {error}") from error
    choices = payload.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content is None:
        # Some servers send the full message on non-streaming fallbacks.
        content = (choices[0].get("message") or {}).get("content")
    return str(content) if content else None


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str = "",
        temperature: float = 0.4,
        max_tokens: int = 300,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not base_url:
            raise LLMError("AGENT_LLM_BASE_URL is required.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_seconds = timeout_seconds

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload_messages = list(messages)
        if not any(message.get("role") == "system" for message in payload_messages):
            payload_messages.insert(0, {"role": "system", "content": self.system_prompt})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=15.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:500]
                        raise LLMError(
                            f"LLM returned HTTP {response.status_code}: {detail}"
                        )
                    async for line in response.aiter_lines():
                        delta = parse_sse_line(line)
                        if delta:
                            yield delta
        except httpx.HTTPError as error:
            raise LLMError(f"LLM request failed: {error}") from error


def build_llm(settings) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        settings.agent_llm_base_url,
        model=settings.agent_llm_model,
        api_key=settings.agent_llm_api_key,
        temperature=settings.agent_llm_temperature,
        max_tokens=settings.agent_llm_max_tokens,
        system_prompt=settings.agent_system_prompt or DEFAULT_SYSTEM_PROMPT,
    )
