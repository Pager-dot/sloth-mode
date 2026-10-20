"""Access to an open-weight model, wherever you keep it.

Four backends. The first two run on your own machine; the other two are for
people who would rather not install or download a model at all.

``ollama``
    The default. Talks to ``/api/chat`` on http://127.0.0.1:11434 and uses
    Ollama's ``format`` field to constrain output to a JSON schema, which is
    what makes 4B-parameter models usable for structured work. Default model:
    ``gemma3:4b``, which handles both text and images.

``openai``
    Any OpenAI-compatible server you run: ``llama.cpp --server``, vLLM,
    LM Studio, text-generation-webui. Uses ``/v1/chat/completions`` and
    ``response_format`` for schema constraints.

``huggingface``
    Hugging Face Inference Providers, which is OpenAI-compatible and has a
    free tier. Runs ``google/gemma-3-4b-it`` without a GPU or a download —
    you need a free token in ``HF_TOKEN``.

``ollama-cloud`` (``cloud``)
    Ollama's hosted catalogue, same ``/api/chat`` wire format as the local
    one but with a bearer token from ``OLLAMA_API_KEY``. Default model
    ``gpt-oss:120b``: Apache-2.0 open weights at a size no laptop will hold,
    and zero disk used here.

The last two *do* send your prompts over the network, which is the trade for
not hosting anything. ``ModelConfig.is_local`` knows the difference and
``sloth doctor`` says which you are on.

Same open weights either way. No SDKs: just the standard library.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Per-backend defaults, applied to any field left empty.
API_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        # Gemma 3 at 4B is multimodal, so one pull covers every agent.
        # `gemma3:1b` is text-only but runs on almost anything.
        "name": "gemma3:4b",
        "vision_name": "gemma3:4b",
    },
    "openai": {
        "base_url": "http://127.0.0.1:8080",
        "name": "local-model",
        "vision_name": "local-model",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "name": "google/gemma-3-4b-it",
        "vision_name": "google/gemma-3-4b-it",
    },
    "ollama-cloud": {
        # Ollama's hosted catalogue: the same wire format as a local Ollama,
        # but the weights are far too big for a laptop and cost no disk here.
        "base_url": "https://ollama.com",
        # Apache-2.0 open weights, 120B parameters, nothing to download.
        "name": "gpt-oss:120b",
        # gpt-oss has no eyes; this is the catalogue's open vision model.
        "vision_name": "gemma4:31b",
    },
}

#: ``--api hf`` is a lot less to type.
API_ALIASES = {
    "hf": "huggingface",
    "openai-compatible": "openai",
    "llamacpp": "openai",
    "cloud": "ollama-cloud",
    "ollama_cloud": "ollama-cloud",
}

#: Backends that need a bearer token, and where each one's token hides.
TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    "ollama-cloud": ("OLLAMA_API_KEY", "OLLAMA_TOKEN"),
}

HF_TOKEN_HELP = (
    "set HF_TOKEN to use Hugging Face. A free account gives you one at "
    "https://huggingface.co/settings/tokens (read access is enough). "
    "Prefer to keep everything on your own machine? "
    "`ollama pull gemma3:4b` and drop the --api flag."
)

OLLAMA_CLOUD_TOKEN_HELP = (
    "set OLLAMA_API_KEY to use Ollama's hosted models. Sign in with "
    "`ollama signin`, or create a key at https://ollama.com/settings/keys. "
    "Prefer to keep everything on your own machine? "
    "`ollama pull gemma3:4b` and drop the --api flag."
)


def token_help(api: str) -> str:
    """How to get a token for ``api``."""
    return {
        "huggingface": HF_TOKEN_HELP,
        "ollama-cloud": OLLAMA_CLOUD_TOKEN_HELP,
    }.get(api, f"{api} needs an API key; set it with --api-key or in config.toml")


class ModelUnavailable(RuntimeError):
    """Raised when the model backend cannot be reached or used."""


class ModelOutputError(RuntimeError):
    """Raised when the model returned something we could not parse."""


@dataclass
class ModelConfig:
    """Which model to use, where it lives, and how patiently to wait for it.

    Every field may be left empty: :meth:`resolved` fills in whatever the
    chosen backend's defaults are, so ``--api hf`` alone is a working config.
    """

    #: Text model. Small is fine; the prompts here are deliberately tiny.
    name: str = ""
    #: Vision model, used only by the fridge agent.
    vision_name: str = ""
    base_url: str = ""
    #: ``"ollama"``, ``"openai"``, or ``"huggingface"``.
    api: str = "ollama"
    temperature: float = 0.0
    timeout: float = 180.0
    num_ctx: int = 8192
    #: Bearer token. Ignored by Ollama; local servers usually ignore it too.
    api_key: str = ""

    def with_overrides(self, **kwargs: Any) -> ModelConfig:
        data = {**self.__dict__, **{k: v for k, v in kwargs.items() if v is not None}}
        return ModelConfig(**data)

    def resolved(self) -> ModelConfig:
        """Return a copy with backend defaults and tokens filled in."""
        api = API_ALIASES.get(self.api.strip().lower(), self.api.strip().lower())
        if api not in API_DEFAULTS:
            raise ValueError(
                f"unknown backend {self.api!r}; try one of: "
                f"{', '.join(sorted(API_DEFAULTS))}"
            )
        defaults = API_DEFAULTS[api]
        key = self.api_key
        if not key:
            variables = TOKEN_VARS.get(api, ())
            key = next((os.environ[v] for v in variables if os.environ.get(v)), "")
        return replace(
            self,
            api=api,
            base_url=self.base_url or defaults["base_url"],
            name=self.name or defaults["name"],
            vision_name=self.vision_name or defaults["vision_name"],
            api_key=key,
        )

    @property
    def dialect(self) -> str:
        """The wire format: ``"ollama"`` (``/api/chat``) or ``"openai"``.

        Ollama's hosted service speaks the same protocol as the one on your
        laptop, so the backend and the dialect are not the same question.
        """
        return "ollama" if self.api in ("ollama", "ollama-cloud") else "openai"

    @property
    def is_local(self) -> bool:
        """True if requests stay on this machine."""
        if self.api in TOKEN_VARS:
            return False
        host = urllib.parse.urlsplit(self.base_url or "").hostname or ""
        return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")

    @property
    def needs_token(self) -> bool:
        return self.api in TOKEN_VARS and not self.api_key


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
    """A thin client for an open-weight model, local by default."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = (config or ModelConfig()).resolved()

    # -- plumbing ---------------------------------------------------------

    def _request(self, path: str, payload: Mapping[str, Any] | None) -> Any:
        config = self.config
        if config.needs_token:
            raise ModelUnavailable(token_help(config.api))

        url = config.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            if config.is_local:
                # Your own machine: ignore any proxy configuration, so a
                # stray HTTPS_PROXY cannot route your files somewhere else.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            else:
                opener = urllib.request.build_opener()
            with opener.open(request, timeout=config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code in (401, 403) and config.api in TOKEN_VARS:
                service = "Hugging Face" if config.api == "huggingface" else "Ollama"
                raise ModelUnavailable(
                    f"{service} rejected the token (HTTP {exc.code}). {detail} "
                    f"{token_help(config.api)}"
                ) from exc
            raise ModelUnavailable(f"{url} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelUnavailable(self._unreachable_message(exc)) from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - needs a server
            raise ModelUnavailable(f"{url} did not return JSON: {exc}") from exc

    def _unreachable_message(self, exc: Exception) -> str:
        where = self.config.base_url
        if self.config.api == "ollama":
            hint = "Is `ollama serve` running?"
        elif self.config.api in TOKEN_VARS:
            hint = "Is this machine online?"
        else:
            hint = "Is your local server running?"
        kind = "a local model" if self.config.is_local else "the model API"
        return f"could not reach {kind} at {where} ({exc}). {hint}"

    # -- capability checks ------------------------------------------------

    def available(self) -> bool:
        """True if this backend looks usable right now.

        For a hosted backend that means "we have a token": listing models over
        the network on every run would make the agents feel slow, and a real
        failure still surfaces at the first call, where the agents handle it
        by falling back to heuristics.
        """
        if self.config.api in TOKEN_VARS:
            return not self.config.needs_token
        try:
            self.list_models()
        except ModelUnavailable:
            return False
        return True

    def list_models(self) -> list[str]:
        """Model names the backend is offering."""
        if self.config.dialect == "ollama":
            payload = self._request("/api/tags", None)
            return sorted(m.get("name", "") for m in payload.get("models", []))
        payload = self._request("/v1/models", None)
        return sorted(m.get("id", "") for m in payload.get("data", []))

    def probe(self) -> str:
        """Make the smallest real request there is, to prove a token works.

        Listing models tells you the service is up, not that it will talk to
        *you*: on a hosted backend the catalogue is usually public while
        inference is not. ``sloth doctor`` uses this so a bad key fails here,
        in a command whose whole job is diagnosis, rather than halfway
        through tidying someone's Downloads folder.
        """
        return self.chat([{"role": "user", "content": "Reply with one word: ok"}])

    def has_model(self, name: str) -> bool:
        """True if ``name`` is available, matching ``gemma3`` to ``gemma3:4b``."""
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
        if self.config.dialect == "ollama":
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
