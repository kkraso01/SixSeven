from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from debate.core.errors import LLMResponseError
from debate.core.protocols import LLMClient

T = TypeVar("T", bound=BaseModel)


class StructuredLLM:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def call(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        max_retries: int = 5,
    ) -> T:
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

                # Add specific guidance for common validation errors
                error_msg = str(exc)

                if "reasons" in error_msg and "too_short" in error_msg:
                    guidance = (
                        "CRITICAL: The 'reasons' field must be a JSON array with SEPARATE items. "
                        'CORRECT format: "reasons": ["First reason", "Second reason", "Third reason"]. '
                        'WRONG format: "reasons": ["First; Second; Third"]. '
                        "Each reason must be a separate string in the array, NOT semicolon-separated in one string. "
                        f"Validation error: {exc}"
                    )
                elif "missing" in error_msg.lower() or "required" in error_msg.lower():
                    guidance = (
                        f"CRITICAL: Your JSON is missing REQUIRED fields. {exc}\n"
                        "You MUST return a FLAT JSON object for ONE agent only — "
                        "do NOT nest under 'CA'/'SA' keys or combine multiple agents.\n"
                        "You MUST include ALL of these fields at the TOP LEVEL:\n"
                        '- "speaker" (string: "CA" or "SA"), "round" (int), '
                        '"tactic_used" (string), "claim" (string), '
                        '"reasons" (array of 2-4 strings), '
                        '"question_to_opponent" (string), "confidence" (int 0-100), '
                        '"what_changes_mind" (string, NOT null), "tone" (string)\n'
                        "If you are the Scientific Agent, you ALSO need: clarify, evaluate_gaps, "
                        "alternative_hypotheses, discriminating_tests\n"
                        "Do NOT omit any field. Return complete valid JSON."
                    )
                elif (
                    "invalid json" in error_msg.lower()
                    or "eof" in error_msg.lower()
                    or "unterminated" in error_msg.lower()
                ):
                    guidance = (
                        "CRITICAL: Your JSON was truncated or malformed. "
                        "You MUST output COMPLETE, VALID JSON that closes all braces and brackets. "
                        "Keep your text values SHORT (under 40 words each) to avoid truncation. "
                        "Do NOT wrap your JSON in markdown code fences like ```json. "
                        "Return ONLY the raw JSON object."
                    )
                else:
                    guidance = f"You MUST output valid JSON for the schema. Validation error: {exc}"

                current_messages = list(current_messages)
                current_messages.append(
                    {
                        "role": "system",
                        "content": guidance,
                    }
                )
            except Exception as exc:
                # Catch any other unexpected error (network, timeout, etc.)
                attempt += 1
                if attempt >= max_retries:
                    raise LLMResponseError(
                        f"Unexpected error after {max_retries} attempts: {exc}", raw_output=""
                    ) from exc
                # No guidance to add — just retry
                continue
        raise RuntimeError("Unreachable retry loop.")
