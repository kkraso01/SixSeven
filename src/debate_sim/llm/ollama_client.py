from __future__ import annotations

import json
import os
import re
import time
from typing import Any, TypeVar

import httpx
import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..core.config import DebateConfig
from ..core.errors import LLMResponseError  # re-export for backwards compat

T = TypeVar("T", bound=BaseModel)

__all__ = ["OllamaClient", "LLMResponseError"]


# Rate-limit keywords used by both the client and the batch runner
_RATE_LIMIT_KEYWORDS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "resource exhausted",
    "resourceexhausted",
    "too many requests",
)

_MAX_RATE_RETRIES = 10  # per single LLM call

#: HTTP timeout (seconds) for raw Ollama chat requests.
_HTTP_TIMEOUT: float = 120.0
#: Read timeout (seconds) for the OpenAI-compatible Ollama client.
_OPENAI_READ_TIMEOUT: float = 180.0
#: Connect timeout (seconds) for the OpenAI-compatible Ollama client.
_OPENAI_CONNECT_TIMEOUT: float = 30.0
#: Minimum max_tokens sent to Gemini (the API requires a reasonable budget).
_GEMINI_MIN_TOKENS: int = 2048
#: Extra seconds added on top of the server-suggested retry delay.
_RATE_LIMIT_EXTRA_DELAY: float = 5.0
#: Fallback delay (seconds) when the server error doesn't suggest one.
_RATE_LIMIT_FALLBACK_DELAY: float = 20.0


class OllamaClient:
    def __init__(self, config: DebateConfig) -> None:
        self.config = config
        self.http = httpx.Client(
            base_url=config.base_url, timeout=_HTTP_TIMEOUT, default_encoding="utf-8"
        )
        self._instructor_client: Any | None = None
        self._instructor_gemini: Any | None = None

        # ── Always set up OpenAI/Ollama instructor client (for non-Gemini models) ──
        self._openai_client = OpenAI(
            base_url=config.base_url,
            api_key="ollama",
            timeout=httpx.Timeout(_OPENAI_READ_TIMEOUT, connect=_OPENAI_CONNECT_TIMEOUT),
            http_client=httpx.Client(
                verify=False,
                timeout=httpx.Timeout(_OPENAI_READ_TIMEOUT, connect=_OPENAI_CONNECT_TIMEOUT),
                default_encoding="utf-8",
            ),
            max_retries=0,  # We handle retries in instructor_wrapper
        )
        self._instructor_client = instructor.from_openai(
            self._openai_client, mode=instructor.Mode.JSON
        )

        # ── Set up Gemini instructor client if API key is available ──
        api_key = (
            config.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if api_key:
            try:
                # Ensure the env var is set for from_provider to pick up
                os.environ["GOOGLE_API_KEY"] = api_key
                self._instructor_gemini = instructor.from_provider(
                    "google/gemini-3-flash-preview",  # default model (overridden per-call)
                    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
                )
                print("  Gemini instructor client ready (via google-genai + instructor)")
            except ImportError as exc:
                raise ImportError(
                    "google-genai package required for Gemini API. "
                    "Install with:  pip install 'instructor[google-genai]'"
                ) from exc

    # ── Raw Ollama HTTP (fallback) ──────────────────────────────────────────

    def _ollama_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        payload: dict[str, Any] = {
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
        content: str = data.get("message", {}).get("content", "")
        return content

    # ── Gemini structured call (instructor + rate-limit retry) ──────────────

    @staticmethod
    def _map_roles_for_gemini(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Gemini uses 'model' role instead of 'assistant'. Map before sending."""
        mapped = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            mapped.append({**msg, "role": role})
        return mapped

    def _gemini_structured_call(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> T:
        """Call Gemini via Instructor with automatic rate-limit backoff."""
        assert self._instructor_gemini is not None
        last_error: Exception | None = None

        # Gemini API uses "model" role, not "assistant"
        messages = self._map_roles_for_gemini(messages)

        for attempt in range(1, _MAX_RATE_RETRIES + 1):
            try:
                result: T = self._instructor_gemini.create(
                    response_model=response_model,
                    messages=messages,
                    model=model,
                    generation_config={
                        "temperature": temperature,
                        "max_tokens": max(max_tokens, _GEMINI_MIN_TOKENS),
                    },
                    max_retries=3,  # instructor-level validation retries
                )
                return result
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(kw in err_str for kw in _RATE_LIMIT_KEYWORDS):
                    # Extract suggested delay from error message
                    match = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE)
                    delay = float(match.group(1)) if match else _RATE_LIMIT_FALLBACK_DELAY
                    wait = delay + _RATE_LIMIT_EXTRA_DELAY
                    print(
                        f"\n  Gemini rate limit (attempt {attempt}/{_MAX_RATE_RETRIES}), "
                        f"sleeping {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    continue
                raise  # non-rate-limit error → propagate immediately

        raise RuntimeError(
            f"Gemini rate limit persisted after {_MAX_RATE_RETRIES} retries: {last_error}"
        )

    # ── Main entry point ───────────────────────────────────────────────────

    def generate(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        validation_hint: str,
    ) -> T:
        # ── Route Gemini models to the Gemini instructor client ──
        if model.startswith("gemini-") and self._instructor_gemini is not None:
            return self._gemini_structured_call(
                response_model=response_model,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # ── Everything else → OpenAI instructor (Ollama) ──
        if self._instructor_client is not None:
            result: T = self._instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                timeout=_OPENAI_READ_TIMEOUT,
            )
            return result

        # ── Fallback: raw Ollama chat + manual JSON parsing ──
        raw = self._ollama_chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        # ── Sanitise raw LLM output before JSON parsing ──
        raw = raw.strip()
        # Strip markdown code fences that LLMs love to add
        if raw.startswith("```"):
            first_nl = raw.find("\n")
            if first_nl != -1:
                raw = raw[first_nl + 1 :]
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3].rstrip()
        if raw.startswith("`") and raw.endswith("`"):
            raw = raw.strip("`").strip()
        # Attempt to extract JSON object if there's preamble text
        brace = raw.find("{")
        last_brace = raw.rfind("}")
        if brace != -1 and last_brace != -1 and brace < last_brace:
            raw = raw[brace : last_brace + 1]

        try:
            return response_model.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMResponseError(f"Schema validation failed: {exc}", raw) from exc
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Invalid JSON for schema {response_model.__name__}. {validation_hint}",
                raw,
            ) from exc
