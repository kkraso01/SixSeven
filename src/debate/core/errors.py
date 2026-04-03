"""Shared exception types for the debate simulator."""

from __future__ import annotations


class LLMResponseError(RuntimeError):
    """Raised when an LLM response fails schema validation or is malformed."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output
