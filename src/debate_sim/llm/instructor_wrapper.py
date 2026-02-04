from __future__ import annotations

from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from .ollama_client import LLMResponseError, OllamaClient


class StructuredLLM:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def call(
        self,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        max_retries: int = 3,
    ) -> BaseModel:
        attempt = 0
        current_messages = list(messages)
        while attempt < max_retries:
            try:
                return self.client.generate(
                    response_model=response_model,
                    messages=current_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    validation_hint=f"Ensure output validates {response_model.__name__}.",
                )
            except LLMResponseError as exc:
                attempt += 1
                if attempt >= max_retries:
                    raise
                current_messages = list(current_messages)
                current_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You MUST output valid JSON for the schema. "
                            f"Validation error: {exc}"),
                    }
                )
        raise RuntimeError("Unreachable retry loop.")
