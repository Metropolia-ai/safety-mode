"""OpenAI-SDK client factory. Any OpenAI-compatible endpoint: Meta Model API, OpenRouter, OpenAI.

Env:
  SM_LLM_BASE_URL    default https://api.meta.ai/v1
  SM_LLM_API_KEY     required for live calls
  SM_LLM_MODEL       default muse-spark-1.3
  SM_LLM_PROVIDER    name for logs; default derived from the base URL host
  SM_LLM_STRUCTURED  auto (default) | json_schema | json_object
  SM_LLM_REASONING   reasoning_effort to send (minimal|low|medium|high|...); unset = provider default
"""
import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import openai
from openai import OpenAI

log = logging.getLogger("safety_mode.llm")

META_BASE_URL = "https://api.meta.ai/v1"
META_MODEL = "muse-spark-1.3"
# A rule is ~300 visible tokens, but on reasoning models (Muse Spark) max_tokens also caps the reasoning:
# at 2000 Spark twice spent the whole budget thinking about r04 and returned content=None. 8000 still stops runaways.
MAX_TOKENS = 8000

# One entry per completed call: {"model", "mode", "prompt_tokens", "completion_tokens"}. Read by the live test.
USAGE: list[dict] = []


def _record(resp, cfg, mode):
    u = getattr(resp, "usage", None)
    det = getattr(u, "completion_tokens_details", None)
    entry = {"model": cfg.model, "mode": mode,
             "prompt_tokens": getattr(u, "prompt_tokens", None) or 0,
             "completion_tokens": getattr(u, "completion_tokens", None) or 0,
             "reasoning_tokens": getattr(det, "reasoning_tokens", None) or 0,
             "finish_reason": getattr(resp.choices[0], "finish_reason", None) if resp.choices else None}
    USAGE.append(entry)
    log.info("%s/%s mode=%s tokens in=%d out=%d (reasoning %d) finish=%s", cfg.provider, cfg.model, mode,
             entry["prompt_tokens"], entry["completion_tokens"], entry["reasoning_tokens"], entry["finish_reason"])


@dataclass
class LLMConfig:
    base_url: str = META_BASE_URL
    api_key: str = ""
    model: str = META_MODEL
    provider: str = ""
    structured: str = "auto"  # auto | json_schema | json_object
    reasoning: str = ""  # reasoning_effort; empty = not sent

    def __post_init__(self):
        if not self.provider:
            host = urlparse(self.base_url).hostname or "unknown"
            self.provider = {"api.meta.ai": "meta", "openrouter.ai": "openrouter",
                             "api.openai.com": "openai"}.get(host, host)


def config_from_env() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ.get("SM_LLM_BASE_URL", META_BASE_URL),
        api_key=os.environ.get("SM_LLM_API_KEY", ""),
        model=os.environ.get("SM_LLM_MODEL", META_MODEL),
        provider=os.environ.get("SM_LLM_PROVIDER", ""),
        structured=os.environ.get("SM_LLM_STRUCTURED", "auto"),
        reasoning=os.environ.get("SM_LLM_REASONING", ""),
    )


def make_client(cfg: LLMConfig) -> OpenAI:
    if not cfg.api_key:
        raise RuntimeError("SM_LLM_API_KEY is not set")
    # SM_LLM_TIMEOUT / SM_LLM_MAX_RETRIES: the web demo on Vercel has 60 s per request
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                  timeout=float(os.environ.get("SM_LLM_TIMEOUT", "90")),
                  max_retries=int(os.environ.get("SM_LLM_MAX_RETRIES", "2")))


def chat_json(client, cfg: LLMConfig, messages, schema, schema_name="safety_rule"):
    """One chat call that should return a JSON object. Returns (text, mode).

    mode json_schema: strict structured output. mode json_object: the provider only promises JSON,
    so the schema goes into the system prompt and the caller's schema validation is the real gate.
    In auto mode a 400/422 on json_schema switches this config to json_object for the rest of the run.
    """
    extra = {"reasoning_effort": cfg.reasoning} if cfg.reasoning else {}
    if cfg.structured in ("auto", "json_schema"):
        try:
            resp = client.chat.completions.create(
                model=cfg.model, messages=messages, temperature=0, max_tokens=MAX_TOKENS, **extra,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": schema_name, "strict": True, "schema": schema}})
            _record(resp, cfg, "json_schema")
            return resp.choices[0].message.content, "json_schema"
        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            if cfg.structured == "json_schema":
                raise
            log.warning("%s/%s rejected json_schema (%s); falling back to json_object",
                        cfg.provider, cfg.model, str(e)[:500])
            cfg.structured = "json_object"
    msgs = [dict(m) for m in messages]
    msgs[0]["content"] += ("\n\n## Output JSON schema\nReturn one JSON object that validates against this schema. "
                           "No prose, no markdown fences.\n```json\n" + json.dumps(schema) + "\n```")
    resp = client.chat.completions.create(model=cfg.model, messages=msgs, temperature=0,
                                          response_format={"type": "json_object"}, max_tokens=MAX_TOKENS, **extra)
    _record(resp, cfg, "json_object")
    return resp.choices[0].message.content, "json_object"
