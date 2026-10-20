"""Access to a locally hosted open-weight model.

Two backends, both of which you run yourself:

``ollama``
    The default. Talks to ``/api/chat`` on http://127.0.0.1:11434 and uses
    Ollama's ``format`` field to constrain output to a JSON schema, which is
    what makes 3B-parameter models usable for structured work.

``openai``
    Any OpenAI-compatible local server: ``llama.cpp --server``, vLLM,
    LM Studio, text-generation-webui. Uses ``/v1/chat/completions`` and
    ``response_format`` for schema constraints.

No SDKs, no API keys, no network egress beyond your own machine.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ModelUnavailable(RuntimeError):
    """Raised when the local model server cannot be reached or used."""


class ModelOutputError(RuntimeError):
    """Raised when the model returned something we could not parse."""


@dataclass
class ModelConfig:
    """Which local model to use, and how patiently to wait for it."""

    #: Text model. Small is fine; the prompts here are deliberately tiny.
    name: str = "llama3.2:3b"
    #: Vision model, used only by the fridge agent.
    vision_name: str = "qwen2.5vl:3b"
    base_url: str = "http://127.0.0.1:11434"
    #: ``"ollama"`` or ``"openai"``.
    api: str = "ollama"
    temperature: float = 0.0
    timeout: float = 180.0
    num_ctx: int = 8192
    #: Only used by the ``openai`` backend; most local servers ignore it.
    api_key: str = "local"

    def with_overrides(self, **kwargs: Any) -> ModelConfig:
        data = {**self.__dict__, **{k: v for k, v in kwargs.items() if v is not None}}
        return ModelConfig(**data)


@runtime_checkable
class Model(Protocol):
    """The whole model interface. Agents depend on this, not on a vendor."""

    def available(self) -> bool: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> str: ...


class LocalModel:
    """A thin client for a model running on your own machine."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    # -- plumbing ---------------------------------------------------------

    def _request(self, path: str, payload: Mapping[str, Any] | None) -> Any:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.config.api == "openai":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            # Local server only; no proxy, and no outbound traffic.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs a server
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ModelUnavailable(f"{url} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelUnavailable(
                f"could not reach a local model at {self.config.base_url} ({exc}). "
                "Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - needs a server
            raise ModelUnavailable(f"{url} did not return JSON: {exc}") from exc

    # -- capability checks ------------------------------------------------

    def available(self) -> bool:
        """True if the server answers and has at least one model pulled."""
        try:
            self.list_models()
        except ModelUnavailable:
            return False
        return True

    def list_models(self) -> list[str]:
        """Names of the models the local server has on disk."""
        if self.config.api == "ollama":
            payload = self._request("/api/tags", None)
            return sorted(m.get("name", "") for m in payload.get("models", []))
        payload = self._request("/v1/models", None)
        return sorted(m.get("id", "") for m in payload.get("data", []))

    def has_model(self, name: str) -> bool:
        """True if ``name`` is pulled, matching ``llama3.2`` to ``llama3.2:3b``."""
        try:
            installed = self.list_models()
        except ModelUnavailable:
            return False
        return any(m == name or m.split(":")[0] == name.split(":")[0] for m in installed)

    # -- inference --------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> str:
        """Run one chat completion and return the assistant's text."""
        model_name = model or self.config.name
        if self.config.api == "ollama":
            return self._chat_ollama(messages, schema, model_name, images)
        return self._chat_openai(messages, schema, model_name, images)

    def json_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        *,
        model: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> Any:
        """Like :meth:`chat`, but parse the reply as JSON."""
        raw = self.chat(messages, schema=schema, model=model, images=images)
        return parse_json(raw)

    def _chat_ollama(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None,
        model_name: str,
        images: Sequence[Path] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": _attach_images(messages, images, style="ollama"),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        if schema:
            payload["format"] = schema
        response = self._request("/api/chat", payload)
        content = (response.get("message") or {}).get("content")
        if not isinstance(content, str):  # pragma: no cover - needs a server
            raise ModelOutputError(f"unexpected Ollama response: {response!r}")
        return content

    def _chat_openai(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any] | None,
        model_name: str,
        images: Sequence[Path] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": _attach_images(messages, images, style="openai"),
            "temperature": self.config.temperature,
            "stream": False,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "sloth", "strict": True, "schema": schema},
            }
        response = self._request("/v1/chat/completions", payload)
        try:  # pragma: no cover - needs a server
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise ModelOutputError(f"unexpected response: {response!r}") from exc


class NullModel:
    """A model that is never available.

    Used by ``--no-model``, so every agent's heuristic path is exercised by
    exactly the same code the real fallback uses.
    """

    def available(self) -> bool:
        return False

    def chat(self, messages, *, schema=None, model=None, images=None) -> str:
        raise ModelUnavailable("running with --no-model; heuristics only")


def _attach_images(
    messages: Sequence[Mapping[str, Any]],
    images: Sequence[Path] | None,
    *,
    style: str,
) -> list[dict[str, Any]]:
    """Attach base64 images to the last user message, per-backend."""
    out = [dict(m) for m in messages]
    if not images:
        return out

    encoded = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images]
    target = next(
        (m for m in reversed(out) if m.get("role") == "user"),
        None,
    )
    if target is None:  # pragma: no cover - agents always send a user turn
        target = {"role": "user", "content": ""}
        out.append(target)

    if style == "ollama":
        target["images"] = encoded
    else:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": target.get("content") or ""}
        ]
        for blob in encoded:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{blob}"},
                }
            )
        target["content"] = parts
    return out


def parse_json(raw: str) -> Any:
    """Parse JSON from a model, tolerating fences and surrounding chatter.

    Schema-constrained output usually makes this unnecessary. Usually.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ModelOutputError(f"model did not return JSON: {str(raw)[:300]!r}")
