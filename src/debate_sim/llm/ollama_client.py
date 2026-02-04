from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

import httpx
import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import DebateConfig


class LLMResponseError(RuntimeError):
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class OllamaClient:
    def __init__(self, config: DebateConfig) -> None:
        self.config = config
        self.http = httpx.Client(base_url=config.base_url, timeout=120.0)
        self._openai_client: Optional[OpenAI] = None
        self._instructor_client: Optional[Any] = None

        if config.api_mode == "openai":
            # Use base_url directly without appending /v1 (it should already be in the base_url)
            self._openai_client = OpenAI(
                base_url=config.base_url,
                api_key="ollama",
                http_client=httpx.Client(verify=False, timeout=120.0)
            )
            self._instructor_client = instructor.from_openai(
                self._openai_client, mode=instructor.Mode.JSON
            )

    def _ollama_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        response = self.http.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def generate(
        self,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        validation_hint: str,
    ) -> BaseModel:
        if self.config.api_mode == "openai":
            assert self._instructor_client is not None
            return self._instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

        raw = self._ollama_chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMResponseError(f"Schema validation failed: {exc}", raw) from exc
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Invalid JSON for schema {response_model.__name__}. {validation_hint}", raw
            ) from exc
