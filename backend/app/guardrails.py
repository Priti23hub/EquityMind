"""Small, explicit guardrails for EquityMind's input, tools, and output.

These checks are deliberately conservative heuristics, not a complete security
boundary. They keep untrusted text away from the router and restrict the tool
surface without changing the LangGraph architecture.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


MAX_INPUT_LENGTH = 8_000
MAX_TOOL_QUERY_LENGTH = 2_000
MAX_OUTPUT_HOLDBACK = 128

ALLOWED_TOOLS = {"search_filings"}

_PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(reveal|show|print|tell me)\s+(the\s+)?(system|developer)\s+prompt",
    r"(jailbreak|prompt injection|bypass\s+(the\s+)?guardrails)",
)
_SECRET_REQUEST_PATTERNS = (
    r"(show|give|tell|reveal|print|expose|extract).{0,40}(api\s*key|secret|password|token)",
    r"(api\s*key|secret|password|token).{0,40}(show|give|tell|reveal|print|expose|extract)",
    r"(openai|aws|azure).{0,20}(api\s*key|secret)",
    r"(read|dump|list|access).{0,30}(\.env|environment variable|os\.environ)",
)
_UNSAFE_PATTERNS = (
    r"(malware|ransomware|credential theft|steal credentials)",
    r"(reverse shell|keylogger|ddos|denial.of.service)",
    r"(delete|wipe|destroy).{0,30}(database|files|system)",
    r"(execute|run).{0,30}(shell command|powershell|arbitrary code)",
)
_UNRELATED_PATTERNS = (
    r"\b(recipe|cooking|meal plan|vacation itinerary|relationship advice)\b",
    r"\b(fantasy football|video game walkthrough|movie recommendation)\b",
)
_SECRET_VALUE_PATTERNS = (
    r"\bsk-[A-Za-z0-9_-]{20,}\b",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b",
    r"\bBearer\s+[A-Za-z0-9._-]{20,}\b",
    r"\b(?:api[_ -]?key|secret|token|password)\s*[:=]\s*[^\s,;]{8,}\b",
)
_PROHIBITED_OUTPUT_PATTERNS = (
    r"\bguaranteed\s+(return|profit|gains?)\b",
    r"\b(you should|you must|I recommend)\s+(buy|sell|short)\b",
    r"\bthis is not financial advice\b.{0,20}\bbut\b.{0,30}\b(buy|sell)\b",
)
_INTERNAL_LEAK_PATTERNS = (
    r"\bOPENAI_API_KEY\b",
    r"\bos\.environ\b|\bprocess\.env\b",
    r"\b(system|developer) prompt\b",
    r"\bTraceback \(most recent call last\)\b",
)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    stage: str = ""

    def as_event(self) -> dict:
        return {
            "type": "guardrail",
            "stage": self.stage,
            "allowed": self.allowed,
            "reason": self.reason,
        }


def validate_input(message: str) -> GuardrailResult:
    """Validate text before it reaches the router or an LLM."""
    if not isinstance(message, str) or not message.strip():
        return GuardrailResult(False, "Please enter a question for EquityMind.", "input")
    if len(message) > MAX_INPUT_LENGTH:
        return GuardrailResult(False, "That message is too long. Please shorten it and try again.", "input")
    if _matches(message, _PROMPT_INJECTION_PATTERNS):
        return GuardrailResult(False, "I can only help with EquityMind research tasks.", "input")
    if _matches(message, _SECRET_REQUEST_PATTERNS):
        return GuardrailResult(False, "I cannot provide keys, secrets, passwords, or environment values.", "input")
    if _matches(message, _UNSAFE_PATTERNS):
        return GuardrailResult(False, "I cannot help with unsafe or destructive instructions.", "input")
    if _matches(message, _UNRELATED_PATTERNS):
        return GuardrailResult(False, "Please keep requests focused on equity research and company filings.", "input")
    return GuardrailResult(True, stage="input")


def validate_tool_request(tool_name: str, query: str) -> GuardrailResult:
    """Allow only named, bounded tool requests; never execute user code."""
    if tool_name not in ALLOWED_TOOLS:
        return GuardrailResult(False, f"Tool '{tool_name}' is not permitted.", "execution")
    if not isinstance(query, str) or not query.strip():
        return GuardrailResult(False, "The filing search query cannot be empty.", "execution")
    if len(query) > MAX_TOOL_QUERY_LENGTH:
        return GuardrailResult(False, "The filing search query is too long.", "execution")
    if _matches(query, _UNSAFE_PATTERNS) or re.search(
        r"(exec|eval|import\s+os|subprocess|shell|powershell|\.env|os\.environ)",
        query,
        flags=re.IGNORECASE,
    ):
        return GuardrailResult(False, "Tool requests cannot execute code or access the host system.", "execution")
    return GuardrailResult(True, stage="execution")


def _sensitive_environment_values() -> set[str]:
    values = set()
    for name, value in os.environ.items():
        if value and len(value) >= 12 and re.search(r"(key|secret|token|password|credential)", name, re.IGNORECASE):
            values.add(value)
    return values


def validate_output(text: str) -> GuardrailResult:
    """Reject obvious secret leakage and directive investment advice."""
    if _matches(text, _SECRET_VALUE_PATTERNS) or any(
        value in text for value in _sensitive_environment_values()
    ):
        return GuardrailResult(False, "The response was blocked because it may contain sensitive information.", "output")
    if _matches(text, _PROHIBITED_OUTPUT_PATTERNS):
        return GuardrailResult(False, "The response was blocked because EquityMind does not provide personalized investment instructions.", "output")
    if _matches(text, _INTERNAL_LEAK_PATTERNS):
        return GuardrailResult(False, "The response was blocked because it may expose internal implementation details.", "output")
    return GuardrailResult(True, stage="output")


class StreamingOutputGuardrail:
    """Inspect streamed text while holding back a small suffix.

    Holding back the longest known pattern window catches a secret or unsafe
    phrase split across token boundaries while preserving token events for the
    rest of the response. The final suffix is released only after validation.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._blocked: Optional[GuardrailResult] = None

    def feed(self, text: str) -> tuple[str, Optional[GuardrailResult]]:
        if self._blocked:
            return "", self._blocked
        self._pending += text
        result = validate_output(self._pending)
        if not result.allowed:
            self._blocked = result
            return "", result
        if len(self._pending) <= MAX_OUTPUT_HOLDBACK:
            return "", None
        safe_text = self._pending[:-MAX_OUTPUT_HOLDBACK]
        self._pending = self._pending[-MAX_OUTPUT_HOLDBACK:]
        return safe_text, None

    def finish(self) -> tuple[str, Optional[GuardrailResult]]:
        if self._blocked:
            return "", self._blocked
        result = validate_output(self._pending)
        if not result.allowed:
            self._blocked = result
            return "", result
        pending, self._pending = self._pending, ""
        return pending, None
