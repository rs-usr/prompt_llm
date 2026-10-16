"""Provider registry and minimal HTTP clients (standard library only)."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 4096


class LLMError(Exception):
    """Raised when a provider request fails."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    default_model: str
    env_var: str = ""  # empty when no API key is needed
    api: str = "openai"  # "openai" (chat/completions) or "anthropic" (messages)
    max_tokens_param: str = "max_tokens"


PROVIDERS = {
    p.name: p
    for p in (
        Provider("openai", "https://api.openai.com/v1", "gpt-5-mini", "OPENAI_API_KEY", max_tokens_param="max_completion_tokens"),
        Provider("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-5-5", "ANTHROPIC_API_KEY", api="anthropic"),
        Provider("google", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", "GOOGLE_API_KEY"),
        Provider("mistralai", "https://api.mistral.ai/v1", "mistral-large-latest", "MISTRAL_API_KEY"),
        Provider("cohere", "https://api.cohere.ai/compatibility/v1", "command-a-03-2025", "COHERE_API_KEY"),
        Provider("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        Provider("deepseek", "https://api.deepseek.com", "deepseek-chat", "DEEPSEEK_API_KEY"),
        Provider("xai", "https://api.x.ai/v1", "grok-4", "XAI_API_KEY"),
        Provider("openrouter", "https://openrouter.ai/api/v1", "openrouter/auto", "OPENROUTER_API_KEY"),
        Provider("together", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_API_KEY"),
        Provider("fireworks", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p3-70b-instruct", "FIREWORKS_API_KEY"),
        Provider("nvidia", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct", "NVIDIA_API_KEY"),
        Provider("ollama", "http://localhost:11434/v1", "llama3.2"),
    )
}

ALIASES = {"claude": "anthropic", "gemini": "google", "mistral": "mistralai", "grok": "xai", "togetherai": "together"}


def get_provider(name: str) -> Provider:
    provider = PROVIDERS.get(ALIASES.get(name, name))
    if provider is None:
        raise LLMError(f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}")
    return provider


@dataclass
class Result:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[dict] = None
    model: Optional[str] = None
    raw: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "text": self.text,
            "tool_calls": self.tool_calls,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


def normalize_tools(data) -> List[dict]:
    """Accept OpenAI-style, Anthropic-style or plain {name, description, parameters} tool definitions."""
    if isinstance(data, dict):
        data = data.get("tools", [data])
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty list of tool definitions")
    tools = []
    for item in data:
        spec = item.get("function") if isinstance(item, dict) and isinstance(item.get("function"), dict) else item
        if not isinstance(spec, dict) or not spec.get("name"):
            raise ValueError(f"every tool needs a 'name': {json.dumps(item)[:200]}")
        tool = {
            "name": spec["name"],
            "parameters": spec.get("parameters") or spec.get("input_schema") or {"type": "object", "properties": {}},
        }
        if spec.get("description"):
            tool["description"] = spec["description"]
        tools.append(tool)
    return tools


def _text(content) -> str:
    # Some providers return content as a list of parts instead of a string.
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _request(provider, messages, *, api_key, model, base_url, system, temperature, max_tokens, tools, stream):
    base = (base_url or provider.base_url).rstrip("/")
    if provider.api == "anthropic":
        url = f"{base}/messages"
        headers = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        payload = {"model": model, "max_tokens": max_tokens or DEFAULT_MAX_TOKENS, "messages": messages}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {**{k: v for k, v in t.items() if k != "parameters"}, "input_schema": t["parameters"]} for t in tools
            ]
    else:
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {"model": model, "messages": ([{"role": "system", "content": system}] if system else []) + messages}
        if max_tokens:
            payload[provider.max_tokens_param] = max_tokens
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
    if temperature is not None:
        payload["temperature"] = temperature
    if stream:
        payload["stream"] = True
    return url, headers, payload


def _open(url: str, headers: dict, payload: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "prompt_llm", **headers},
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(body)
            if isinstance(error, list) and error:
                error = error[0]
            detail = error.get("error", error)
            message = detail.get("message", body) if isinstance(detail, dict) else detail
        except (ValueError, AttributeError):
            message = body
        raise LLMError(f"HTTP {e.code}: {str(message).strip() or e.reason}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"Connection error: {e.reason}") from None
    except (TimeoutError, socket.timeout):
        raise LLMError(f"Request timed out after {timeout}s") from None
    except OSError as e:
        raise LLMError(f"Connection error: {e}") from None


def _parse(provider: Provider, data: dict) -> Result:
    if provider.api == "anthropic":
        blocks = data.get("content") or []
        usage = data.get("usage")
        return Result(
            text="".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
            tool_calls=[
                {"id": b.get("id"), "name": b.get("name"), "arguments": b.get("input", {})}
                for b in blocks
                if b.get("type") == "tool_use"
            ],
            stop_reason=data.get("stop_reason"),
            usage={"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")} if usage else None,
            model=data.get("model"),
            raw=data,
        )

    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected response: {json.dumps(data)[:500]}") from None
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass  # keep the raw string if the model produced invalid JSON
        tool_calls.append({"id": call.get("id"), "name": function.get("name"), "arguments": arguments})
    usage = data.get("usage")
    return Result(
        text=_text(message.get("content")),
        tool_calls=tool_calls,
        stop_reason=choice.get("finish_reason"),
        usage={"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")} if usage else None,
        model=data.get("model"),
        raw=data,
    )


def complete(
    provider: Provider,
    messages: List[dict],
    *,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[dict]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result:
    """Send messages and return the full, parsed response."""
    url, headers, payload = _request(
        provider, messages, api_key=api_key, model=model, base_url=base_url, system=system,
        temperature=temperature, max_tokens=max_tokens, tools=tools, stream=False,
    )
    with _open(url, headers, payload, timeout) as response:
        try:
            body = response.read().decode("utf-8", errors="replace")
        except (TimeoutError, socket.timeout):
            raise LLMError(f"Request timed out after {timeout}s") from None
    try:
        data = json.loads(body)
    except ValueError:
        raise LLMError(f"Invalid JSON response: {body[:500]}") from None
    return _parse(provider, data)


def stream(
    provider: Provider,
    messages: List[dict],
    *,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[str]:
    """Send messages and yield response text as it arrives (server-sent events)."""
    url, headers, payload = _request(
        provider, messages, api_key=api_key, model=model, base_url=base_url, system=system,
        temperature=temperature, max_tokens=max_tokens, tools=None, stream=True,
    )
    with _open(url, headers, payload, timeout) as response:
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if "error" in event:
                    error = event["error"]
                    raise LLMError(error.get("message", data) if isinstance(error, dict) else str(error))
                if provider.api == "anthropic":
                    text = (event.get("delta") or {}).get("text") if event.get("type") == "content_block_delta" else None
                else:
                    choices = event.get("choices") or []
                    text = _text((choices[0].get("delta") or {}).get("content")) if choices else None
                if text:
                    yield text
        except (TimeoutError, socket.timeout):
            raise LLMError(f"Stream timed out after {timeout}s") from None
        except OSError as e:
            raise LLMError(f"Connection error: {e}") from None
