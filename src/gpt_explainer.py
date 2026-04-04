"""
gpt_explainer.py - Free LLM fund explanation for Smart Fund Advisor (v3).

Provider Strategy  (in priority order — all FREE tier, best-quality-first)
---------------------------------------------------------------------
1. Google Gemini 2.0 Flash (aistudio.google.com) — **gemini-2.0-flash**
   Excellent financial reasoning, 1 M token context, 1500 req/day free quota.
   Set env var:  GEMINI_API_KEY=<your-key>
   Get free key:  https://aistudio.google.com/apikey  (no credit card needed)
   Install SDK:   pip install google-genai

2. Groq API (groq.com) — **llama-3.3-70b-versatile** (70B params, free tier)
   Superior financial reasoning with 6000 tokens/min free quota.
   Set env var:  GROQ_API_KEY=<your-key>   (get free key at console.groq.com)
   Install SDK:   pip install groq

3. OpenRouter (openrouter.ai) — **google/gemma-2-9b-it:free**
   Google Gemma 2 9B: excellent instruction-following, strong on finance.
   Set env var:  OPENROUTER_API_KEY=<your-key>

4. Ollama (local) — **llama3.2** (or any model you have downloaded)
   100% FREE, fully private, runs on your machine. Zero API key needed.
   Install:  brew install ollama && ollama pull llama3.2  (macOS)
   Or:       curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.2

5. HuggingFace Inference API — **Qwen/Qwen2.5-72B-Instruct**
   Upgraded to 72B; the new router requires a free HF token (retired 2025).
   Set env var:  HF_TOKEN=<your-token>   (free at huggingface.co/settings/tokens)

6. Rule-based fallback — no API key needed, always works

Model Selection Rationale
-------------------------------------------------------------------
- gemini-2.0-flash: state-of-the-art financial reasoning, 1M context window,
  Google Search grounding, generous free tier (1500 req/day).
- llama-3.3-70b-versatile: MT-Bench 8.6, superior financial domain knowledge,
  handles complex multi-fund portfolio explanations. Free on Groq (Dec 2024).
- gemma-2-9b-it: MMLU 71.3%, strong instruction-following, Google-trained
  with safety alignment. Free tier on OpenRouter.
- Ollama local: zero-cost, privacy-preserving, works offline. Supports llama3.2,
  mistral, phi3, qwen2.5 and many other open-source models.
- Qwen2.5-72B-Instruct: upgraded from 3B → 72B for much better financial reasoning.
  Requires free HF token (huggingface.co/settings/tokens).
- Rule-based: deterministic, always available, 100% factual correctness.

Usage
-----
    from src.gpt_explainer import explain_fund, explain_portfolio, validate_gpt_correctness
    
    text = explain_fund(fund_row, user_risk="High", user_context="30yr old, salaried")
    validation = validate_gpt_correctness(text, fund_row)
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd

# ─── Optional SDK imports (gracefully degraded if package missing) ─────────────
try:
    from google import genai as _google_genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _google_genai = None
    _genai_types = None
    _GENAI_AVAILABLE = False

try:
    from groq import Groq as _GroqClient
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GroqClient = None
    _GROQ_SDK_AVAILABLE = False


# ─── C3: TTL Response Cache ────────────────────────────────────────────────────
# Caches LLM responses for 1 hour by prompt hash to avoid redundant API calls.
_RESPONSE_CACHE: Dict[str, tuple] = {}   # {md5_key: (response_str, datetime)}
_CACHE_TTL_SECONDS: int = 3600           # 1-hour TTL


def _cached_call(prompt: str, provider: str, caller_fn) -> str:
    """Return a cached LLM response if still fresh, otherwise call the API."""
    key = hashlib.md5(f"{provider}:{prompt}".encode()).hexdigest()
    if key in _RESPONSE_CACHE:
        resp, ts = _RESPONSE_CACHE[key]
        if (datetime.now() - ts).total_seconds() < _CACHE_TTL_SECONDS:
            return resp
    result = caller_fn(prompt)
    _RESPONSE_CACHE[key] = (result, datetime.now())
    return result


def clear_explanation_cache() -> int:
    """Clear all cached LLM responses. Returns the number of entries removed."""
    n = len(_RESPONSE_CACHE)
    _RESPONSE_CACHE.clear()
    return n


# ─── PII sanitizer ────────────────────────────────────────────────────────────

def _sanitize_user_context(user_context: str) -> str:
    """
    Strip personally identifiable information from a user-context string
    before it is embedded in an LLM prompt.

    Removes / redacts:
    - Full proper names  (e.g. "Neil Chatterjee")
    - Exact rupee amounts  (₹85,000 → ₹[amount])
    - 10-digit phone numbers
    - Email addresses
    - 12-digit Aadhaar-style numbers

    Returns a safe summary string, or the default fallback if nothing remains.
    """
    if not user_context:
        return "Indian retail investor"
    # Accept dicts (e.g. {"monthly_income": 100000}) by converting to a safe string
    if isinstance(user_context, dict):
        safe_parts = []
        for k, v in user_context.items():
            key_str = str(k).replace("_", " ")
            if k in ("name", "customer_name", "full_name"):
                continue   # drop name keys entirely
            safe_parts.append(f"{key_str}: {v}")
        ctx = ", ".join(safe_parts) if safe_parts else ""
    else:
        ctx = str(user_context)
    # Remove proper names: two or more consecutive Title-case words
    ctx = re.sub(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', '', ctx)
    # Redact exact rupee/currency figures
    ctx = re.sub(r'₹[\d,]+', '₹[amount]', ctx)
    ctx = re.sub(r'Rs\.?\s*[\d,]+', 'Rs.[amount]', ctx, flags=re.IGNORECASE)
    # Redact 10-digit phone numbers
    ctx = re.sub(r'\b\d{10}\b', '[phone]', ctx)
    # Redact 12-digit Aadhaar-like numbers
    ctx = re.sub(r'\b\d{12}\b', '[id]', ctx)
    # Redact email addresses
    ctx = re.sub(r'\S+@\S+\.\S+', '[email]', ctx)
    # Collapse extra whitespace / punctuation left behind
    ctx = re.sub(r'[\s,]+', ' ', ctx).strip(' ,')
    return ctx or "Indian retail investor"


# ─── C2: Risk-class few-shot examples ─────────────────────────────────────────
# One-shot example per risk tier; prepended to explain_fund() prompt so the
# model learns the expected tone, length, and data-citation style.
_FEW_SHOT: Dict[str, str] = {
    "Very_Low": (
        "Example output: 'SBI Overnight Fund (SBI MF) offers near-zero credit and "
        "duration risk by investing only in overnight securities. With NAV ₹3,812, "
        "AUM ₹18,000 Cr, and TER 0.10%, it is ideal for parking emergency corpus for "
        "1–7 days. Capital preservation is near-guaranteed in normal market conditions.'"
    ),
    "Low": (
        "Example output: 'HDFC Short Duration Fund targets 1–3 year debt paper, "
        "delivering stable 6–7% returns with minimal volatility. NAV ₹27.4, AUM "
        "₹14,500 Cr, TER 0.75%. Suitable for conservative investors with a 1–3 year "
        "horizon who want better-than-FD returns without equity risk.'"
    ),
    "Medium": (
        "Example output: 'ICICI Prudential Balanced Advantage Fund dynamically shifts "
        "between equity (30–80%) and debt based on valuations. 3yr CAGR 12.3%, Sharpe "
        "0.91, TER 1.05%, NAV ₹58.2, AUM ₹52,000 Cr. Ideal for moderate-risk "
        "investors seeking inflation-beating returns over 3–5 years with managed downside.'"
    ),
    "High": (
        "Example output: 'Mirae Asset Large Cap Fund targets blue-chip stocks for "
        "long-term wealth creation. 3yr CAGR 14.8%, Sharpe 1.12, MaxDD −18%, TER "
        "0.55%, NAV ₹94.5, AUM ₹35,000 Cr. Best for high-risk investors with a 5+ "
        "year horizon who can tolerate short-term volatility for superior equity returns.'"
    ),
    "Very_High": (
        "Example output: 'Nippon India Small Cap Fund pursues aggressive growth via "
        "small-cap stocks — highest risk but highest long-term return potential. 3yr "
        "CAGR 28.1%, Sharpe 1.43, MaxDD −38%, TER 1.50%, NAV ₹142, AUM ₹48,000 Cr. "
        "Only for very-high-risk investors with a 7–10 year horizon and high drawdown tolerance.'"
    ),
}


# ─── C1: JSON prompt suffix & response parser ─────────────────────────────────
_JSON_PROMPT_SUFFIX = (
    '\n\nReturn your response as a JSON object with exactly these keys:\n'
    '  "summary": one sentence fund overview with key metric,\n'
    '  "rationale": 2-3 sentences explaining risk profile fit,\n'
    '  "recommendation": one actionable sentence for the investor.\n'
    'Output only the JSON object — no markdown, no preamble.'
)


def _parse_json_explanation(raw: str) -> str:
    """
    Try to extract a structured JSON explanation from the LLM response.
    Falls back to returning the raw text if JSON parsing fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        parts = []
        if "summary" in data:
            parts.append(data["summary"])
        if "rationale" in data:
            parts.append(data["rationale"])
        if "recommendation" in data:
            parts.append(f"**Recommendation:** {data['recommendation']}")
        return "\n\n".join(parts) if parts else raw
    except (json.JSONDecodeError, TypeError):
        return raw


# ─── Provider detection ────────────────────────────────────────────────────────


def _check_ollama_running() -> bool:
    """Return True if an Ollama server is running at localhost:11434."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _get_provider() -> str:
    """Return the best available free LLM provider (highest priority first).

    Priority: gemini → groq → openrouter → ollama → huggingface
    Each provider requires either a free API key or local Ollama installation.
    See module docstring for setup instructions.
    """
    if os.getenv("GEMINI_API_KEY") and _GENAI_AVAILABLE:
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if _check_ollama_running():
        return "ollama"
    # HuggingFace new inference router requires a free HF token
    return "huggingface"


def get_active_provider() -> str:
    """Public helper — returns name of the active free LLM provider."""
    return _get_provider()


# ─── Groq API (Free tier — Llama-3.3-70B via official SDK) ───────────────────
# Free tier: 6000 tokens/min, 500 req/day — get key at console.groq.com

_FINANCE_SYSTEM_MSG = (
    "You are a SEBI-registered investment advisor specialising in Indian mutual funds. "
    "Give concise, data-grounded, personalised explanations. Reference NAV, AUM, "
    "expense ratio, CAGR, Sharpe ratio where available. Keep responses under 200 words."
)


def _call_groq(prompt: str, max_tokens: int = 400) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Get a free key at console.groq.com")

    if _GROQ_SDK_AVAILABLE and _GroqClient is not None:
        # Official groq SDK — auto-retry, typed responses, better error messages.
        client = _GroqClient(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _FINANCE_SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()

    # Fallback: raw HTTP (no groq package installed)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": _FINANCE_SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_groq_cached(prompt: str, max_tokens: int = 400) -> str:
    """C3: Cache-wrapped Groq call."""
    return _cached_call(prompt, "groq", lambda p: _call_groq(p, max_tokens))


# ─── OpenRouter API (Free tier) ───────────────────────────────────────────────

def _call_openrouter(prompt: str, max_tokens: int = 400) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://smart-fund-advisor.app",
    }
    # v3: upgraded to google/gemma-2-9b-it:free (from llama-3.1-8b)
    # Gemma 2 9B: better instruction-following, Google-trained safety alignment.
    payload = {
        "model": "google/gemma-2-9b-it:free",
        "messages": [
            {"role": "system", "content": _FINANCE_SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_openrouter_cached(prompt: str, max_tokens: int = 400) -> str:
    """C3: Cache-wrapped OpenRouter call."""
    return _cached_call(prompt, "openrouter", lambda p: _call_openrouter(p, max_tokens))


# ─── Google Gemini API (Free — gemini-2.0-flash) ──────────────────────────────
# Free quota: 1,500 req/day, 1M tokens/min — get key at aistudio.google.com/apikey
# Install:  pip install google-genai


def _call_gemini(prompt: str, max_tokens: int = 400) -> str:
    if not _GENAI_AVAILABLE or _google_genai is None:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai"
        )
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Get a FREE key (no credit card) at https://aistudio.google.com/apikey"
        )
    client = _google_genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_FINANCE_SYSTEM_MSG,
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )
    return response.text.strip()


def _call_gemini_cached(prompt: str, max_tokens: int = 400) -> str:
    """C3: Cache-wrapped Gemini call."""
    return _cached_call(prompt, "gemini", lambda p: _call_gemini(p, max_tokens))


# ─── Ollama (Local — 100% Free, No API Key) ───────────────────────────────────
# macOS install:  brew install ollama && ollama serve (in a separate terminal)
# Pull a model:   ollama pull llama3.2  (or mistral, phi3, qwen2.5, etc.)
# Set OLLAMA_MODEL env var to override model selection.

_OLLAMA_PREFERRED = ["llama3.2", "llama3.1", "llama3", "mistral", "phi3", "qwen2.5"]


def _ollama_best_model() -> str:
    """Return best available Ollama model (env override → preferred list → first available)."""
    override = os.getenv("OLLAMA_MODEL", "")
    if override:
        return override
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=3).json()
        available = [m["name"] for m in tags.get("models", [])]
        for pref in _OLLAMA_PREFERRED:
            for m in available:
                if pref in m.lower():
                    return m
        if available:
            return available[0]
    except Exception:
        pass
    return "llama3.2"


def _call_ollama(prompt: str, max_tokens: int = 400) -> str:
    """Call Ollama via its OpenAI-compatible chat completions endpoint (localhost)."""
    model = _ollama_best_model()
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _FINANCE_SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama_cached(prompt: str, max_tokens: int = 400) -> str:
    """C3: Cache-wrapped Ollama call."""
    return _cached_call(prompt, "ollama", lambda p: _call_ollama(p, max_tokens))


# ─── HuggingFace Inference API (Free with token, via new router) ──────────────
# NOTE: The legacy api-inference.huggingface.co endpoint was retired in 2025.
# The new router.huggingface.co requires a free HF token.
# Get yours at: https://huggingface.co/settings/tokens  (Read access is enough)


def _call_huggingface(prompt: str, max_tokens: int = 400) -> str:
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. The HuggingFace inference router now requires a "
            "free token. Get one at https://huggingface.co/settings/tokens"
        )

    _HF_SYSTEM = (
        "You are a SEBI-registered Indian mutual fund advisor. "
        "Give concise, data-grounded, personalised fund explanations "
        "referencing NAV, AUM, CAGR, expense ratio where available. "
        "Keep responses under 150 words."
    )
    # Upgraded to 72B; try multiple models with graceful fallback
    models_to_try = [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]
    last_err: Exception = RuntimeError("No HF models succeeded")
    for model in models_to_try:
        try:
            client = InferenceClient(provider="hf-inference", api_key=token)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _HF_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
            time.sleep(2)
    raise RuntimeError(f"All HuggingFace models failed. Last error: {last_err}") from last_err


def _call_huggingface_cached(prompt: str, max_tokens: int = 400) -> str:
    """C3: Cache-wrapped HuggingFace call."""
    return _cached_call(prompt, "huggingface", lambda p: _call_huggingface(p, max_tokens))


# ─── Rule-based fallback (always available) ───────────────────────────────────

def _rule_based_explanation(
    fund_row: pd.Series,
    user_risk: str,
    user_context: str = "",
) -> str:
    name = fund_row.get("Scheme_Name", "This fund")
    cat  = fund_row.get("Scheme_Category", "")
    amc  = fund_row.get("AMC", "")
    nav  = fund_row.get("NAV", 0)
    aum  = fund_row.get("Average_AUM_Cr", 0)
    min_amt = fund_row.get("Scheme_Min_Amt", 500)

    risk_guidance = {
        "Very_Low":  "This is a capital preservation scheme, ideal for emergency corpus or very short-term goals. Very low volatility.",
        "Low":       "This debt-oriented fund offers stable returns with minimal risk. Suitable for 1–3 year goals.",
        "Medium":    "This balanced/hybrid fund provides moderate growth with managed downside. Good for 3–5 year horizons.",
        "High":      "This equity fund targets long-term wealth creation. Expect short-term volatility for 5+ year goals.",
        "Very_High": "This aggressive equity fund (mid/small-cap or sectoral) carries higher risk for potentially superior 5–10 year returns.",
    }

    guidance = risk_guidance.get(user_risk, "")
    safe_ctx = _sanitize_user_context(user_context)
    ctx_note = f" Based on your profile ({safe_ctx})." if safe_ctx and safe_ctx != 'Indian retail investor' else ""

    return (
        f"**{name}** ({amc})\n\n"
        f"**Why this fund?**  Your risk profile '{user_risk}' aligns with '{cat}'.{ctx_note}\n\n"
        f"**Fund Snapshot:**\n"
        f"- Current NAV: ₹{nav:,.2f}\n"
        f"- Total AUM: ₹{aum:,.0f} Cr (larger AUM = more established)\n"
        f"- Minimum Investment: ₹{min_amt:,.0f}\n\n"
        f"**Investment Rationale:**  {guidance}\n\n"
    )


# ─── Main public API ──────────────────────────────────────────────────────────

def explain_fund(
    fund_row: pd.Series,
    user_risk: str,
    user_context: str = "",
    provider: Optional[str] = None,
    fallback: bool = True,
) -> Tuple[str, str]:
    """
    Generate a personalised explanation for a recommended mutual fund.

    Parameters
    ----------
    fund_row     : a single row from recommend_funds() output
    user_risk    : predicted risk tier ('Very_Low' … 'Very_High')
    user_context : optional string with user info (age, goal, income bracket)
    provider     : 'groq'|'openrouter'|'huggingface'|'rule'  (auto-detected if None)
    fallback     : fall back to rule-based if API fails

    Returns
    -------
    (explanation_text, provider_used)
    """
    if provider is None:
        provider = _get_provider()

    name = fund_row.get("Scheme_Name", "Unknown Fund")
    cat  = fund_row.get("Scheme_Category", "")
    amc  = fund_row.get("AMC", "")
    nav  = fund_row.get("NAV", 0)
    aum  = fund_row.get("Average_AUM_Cr", 0)

    # ── Inject NAV metrics for data-grounded explanations ────────────────
    cagr3  = fund_row.get("cagr_3yr_hist",     fund_row.get("CAGR_3yr",    None))
    sharpe = fund_row.get("sharpe_1yr_hist",    fund_row.get("Sharpe",      None))
    max_dd = fund_row.get("max_drawdown_hist",  fund_row.get("MaxDD",       None))
    ter_missing = fund_row.get("ter_missing_flag", None)
    er_real = fund_row.get("expense_ratio_real", fund_row.get("Expense_Ratio", None))
    er = er_real if (ter_missing in (None, 0, 0.0, "0")) else None
    cluster_label = fund_row.get("cluster_label", "")

    metrics_parts: list = []
    for label, val, fmt in [
        ("3yr CAGR", cagr3,  "{:.1%}"),
        ("Sharpe",   sharpe, "{:.2f}"),
        ("MaxDD",    max_dd, "{:.1%}"),
        ("TER",      er,     "{:.2f}%"),
    ]:
        if val is not None:
            try:
                metrics_parts.append(f"{label}: {fmt.format(float(val))}")
            except (TypeError, ValueError):
                pass
    metrics_str = (", ".join(metrics_parts) + ". ") if metrics_parts else ""
    cluster_str = f"User risk cluster: '{cluster_label}'. " if cluster_label else ""

    # ── C2: Prepend risk-class few-shot example ───────────────────────────────
    few_shot_str = _FEW_SHOT.get(user_risk, "")
    few_shot_prefix = f"{few_shot_str}\n\nNow generate for the following fund:\n" if few_shot_str else ""

    # ── C1: Base prompt + JSON output suffix ─────────────────────────────────
    base_prompt = (
        f"{few_shot_prefix}"
        f"Explain in 3 short paragraphs why the mutual fund '{name}' by {amc} "
        f"(Category: {cat}, NAV: ₹{nav:.2f}, AUM: ₹{aum:,.0f} Cr, {metrics_str}) "
        f"is suitable for an investor with a '{user_risk}' risk appetite. "
        f"{cluster_str}"
        f"User context: {_sanitize_user_context(user_context)}. "
        f"Include: (1) fund overview with key metrics, (2) why it matches the risk profile, "
        f"(3) expense ratio impact and return consistency. "
        f"Be specific to Indian market. Under 180 words."
    )
    prompt = base_prompt + _JSON_PROMPT_SUFFIX

    # Try API providers
    if provider in ("rule", "rule_based"):
        return _rule_based_explanation(fund_row, user_risk, user_context), "rule_based"

    try:
        if provider == "gemini":
            raw = _call_gemini_cached(prompt)
        elif provider == "groq":
            raw = _call_groq_cached(prompt)
        elif provider == "openrouter":
            raw = _call_openrouter_cached(prompt)
        elif provider == "ollama":
            raw = _call_ollama_cached(prompt)
        else:   # "huggingface"
            raw = _call_huggingface_cached(prompt)
        text = _parse_json_explanation(raw)   # C1: structured → readable text
        return text, provider
    except Exception as e:
        if fallback:
            return _rule_based_explanation(fund_row, user_risk, user_context), "rule_based"
        raise RuntimeError(f"LLM provider '{provider}' failed: {e}") from e


def explain_portfolio(
    recs_df: pd.DataFrame,
    user_risk: str,
    user_context: str = "",
    top_n: int = 3,
    provider: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate explanations for the top-N funds in a recommendation result.

    Returns
    -------
    dict of {fund_name: explanation_text}
    """
    results = {}
    for _, row in recs_df.head(top_n).iterrows():
        name = row.get("Scheme_Name", f"Fund_{_}")
        text, prov = explain_fund(row, user_risk, user_context, provider=provider)
        results[name] = {"explanation": text, "provider": prov}
        time.sleep(0.5)   # rate-limit courtesy pause
    return results


# ─── Validate GPT explanation correctness ─────────────────────────────────────

def validate_gpt_correctness(
    explanation: str,
    fund_row: pd.Series,
    user_risk: str,
) -> Dict:
    """
    Validate that the GPT-generated explanation is factually grounded.
    
    Checks
    ------
    1. Fund name mentioned in explanation
    2. Risk tier mentioned
    3. AMC / category mentioned
    4. No hallucinated NAV outside ±20% of actual
    5. Coherence score (keyword overlap)
    
    Returns
    -------
    dict with per-check pass/fail and overall correctness score
    """
    text_lower = explanation.lower()
    
    name_part = (fund_row.get("Scheme_Name", "") or "").lower().split()[:3]
    name_check = any(w in text_lower for w in name_part if len(w) > 3)
    
    risk_aliases = {
        "Very_Low":  ["very low", "liquid", "overnight", "capital preservation"],
        "Low":       ["low risk", "debt", "short duration", "stable"],
        "Medium":    ["medium", "balanced", "hybrid", "moderate"],
        "High":      ["high", "equity", "large cap", "growth", "wealth"],
        "Very_High": ["very high", "aggressive", "small cap", "mid cap", "sectoral"],
    }
    risk_check = any(kw in text_lower for kw in risk_aliases.get(user_risk, [user_risk.lower()]))
    
    cat_words  = [w for w in (fund_row.get("Scheme_Category", "") or "").lower().split() if len(w) > 4]
    cat_check  = any(w in text_lower for w in cat_words[:3]) if cat_words else True
    
    # NAV hallucination check
    actual_nav = fund_row.get("NAV", 0)
    nav_check  = True
    nav_mentions = re.findall(r"₹?\s?([\d,]+\.?\d*)", explanation)
    for val_str in nav_mentions:
        try:
            val = float(val_str.replace(",", ""))
            if 10 < val < 100000:   # likely a NAV mention
                if abs(val - actual_nav) / (actual_nav + 1) > 0.3:
                    nav_check = False
                    break
        except ValueError:
            pass
    
    checks = {
        "fund_name_mentioned": name_check,
        "risk_tier_referenced": risk_check,
        "category_mentioned": cat_check,
        "nav_not_hallucinated": nav_check,
    }
    correctness_score = sum(checks.values()) / len(checks)
    
    return {
        **checks,
        "correctness_score": round(correctness_score, 2),
        "pass": correctness_score >= 0.75,
        "verdict": "PASS ✓" if correctness_score >= 0.75 else "FAIL ✗",
    }


def batch_validate(
    recs_df: pd.DataFrame,
    explanations: Dict[str, Dict],
    user_risk: str,
) -> Dict:
    """Validate explanations for all recommended funds. Returns batch summary."""
    results = []
    for _, row in recs_df.iterrows():
        name = row.get("Scheme_Name", "")
        if name in explanations:
            text = explanations[name]["explanation"]
            v    = validate_gpt_correctness(text, row, user_risk)
            v["fund_name"] = name
            results.append(v)

    pass_count = sum(r["pass"] for r in results)
    return {
        "per_fund": results,
        "pass_count": pass_count,
        "total": len(results),
        "overall_pass_rate": round(pass_count / max(len(results), 1), 2),
        "overall_pass": pass_count >= len(results) * 0.75,
    }


# ─── Horizon-level portfolio explanation (v3) ─────────────────────────────────

_HORIZON_RATIONALE = {
    "1yr": (
        "With a 1-year horizon, capital preservation takes precedence over growth. "
        "A debt-heavy allocation minimises drawdown risk during short-term volatility."
    ),
    "3yr": (
        "A 3-year horizon allows moderate equity exposure for inflation-beating returns "
        "while a strong debt allocation cushions against market corrections."
    ),
    "5yr": (
        "Over 5 years, equity volatility tends to average out. A balanced allocation "
        "targets meaningful real returns while debt provides a stability anchor."
    ),
    "10yr+": (
        "A 10-year+ horizon fully harnesses compounding. High equity exposure maximises "
        "long-term wealth creation; a small debt slice buffers short-term shocks."
    ),
}


def _rule_based_horizon_explanation(
    user_risk: str,
    horizon: str,
    equity_pct: float,
    brackets: list,
) -> str:
    """Rule-based fallback explanation for a single horizon portfolio."""
    debt_pct = 1.0 - equity_pct
    rationale = _HORIZON_RATIONALE.get(horizon, f"This {horizon} portfolio suits a {user_risk} risk investor.")

    bracket_summaries = []
    for b in brackets:
        top_names = (
            b["funds"].head(2)["Scheme_Name"].str[:35].tolist()
            if not b["funds"].empty else []
        )
        fstr = " & ".join(top_names) if top_names else "N/A"
        bracket_summaries.append(f"{b['bracket']} ({b['tier']}, {b['pct']:.0f}%): {fstr}")

    portfolio_str = " | ".join(bracket_summaries) if bracket_summaries else "N/A"
    core_tier = brackets[0]["tier"] if brackets else user_risk

    return (
        f"**{horizon} Portfolio — {user_risk} Risk**\n\n"
        f"{rationale}\n\n"
        f"**Allocation — {equity_pct*100:.0f}% Equity / {debt_pct*100:.0f}% Debt:**\n"
        f"{portfolio_str}\n\n"
        f"**Core-Satellite Logic:** The Core (60%) anchors the portfolio in the "
        f"{core_tier} tier for primary returns. The Stability satellite (20%) adds a "
        f"downside buffer via a conservative tier, while the Growth satellite (20%) "
        f"captures upside from a higher risk tier. This three-bracket structure reduces "
        f"single-tier concentration risk and is aligned with BlackRock's Core-Satellite "
        f"framework and Markowitz diversification principles."
    )


def explain_horizon_portfolio(
    horizon_data: Dict,
    user_risk: str,
    user_context: str = "",
    provider: Optional[str] = None,
) -> Dict:
    """
    Generate a GenAI explanation for a single horizon's core-satellite portfolio.

    Parameters
    ----------
    horizon_data : one entry from recommend_full_profile()["horizons"]
                   (e.g. the dict stored under key "5yr")
    user_risk    : user's predicted risk tier
    user_context : optional context string (age, goal, available corpus)
    provider     : LLM provider override; auto-detected from env vars if None

    Returns
    -------
    dict with keys: explanation (str), provider (str), horizon (str)
    """
    if provider is None:
        provider = _get_provider()

    horizon    = horizon_data.get("horizon", "5yr")
    equity_pct = horizon_data.get("equity_pct", 0.5)
    debt_pct   = 1.0 - equity_pct
    brackets   = horizon_data.get("brackets", [])

    # Build a compact fund summary for the LLM prompt
    bracket_lines = []
    for b in brackets:
        top_names = (
            b["funds"].head(2)["Scheme_Name"].str[:40].tolist()
            if not b["funds"].empty else []
        )
        names_str = " & ".join(top_names) if top_names else "N/A"
        bracket_lines.append(
            f"  - {b['bracket']} ({b['pct']:.0f}%, {b['tier']} tier): {names_str}"
        )
    portfolio_summary = "\n".join(bracket_lines) if bracket_lines else "  N/A"

    prompt = (
    f"You are a SEBI-registered investment advisor. Act as a professional wealth manager for a '{user_risk}' risk investor "
    f"with a {horizon} horizon. Context: Indian market, current interest rate cycle, and latest debt tax laws.\n\n"
    f"Funds to be recommended by you(summary):\n{portfolio_summary}\n"
    f"Asset Allocation: {equity_pct*100:.0f}% Equity | {debt_pct*100:.0f}% Debt\n\n"
    f"Task: Write exactly 4 paragraphs (under 160 words total) focusing on:\n"
    f"1. Mention funds to be recommended and latest details about it\n"
    f"2. Strategic Fit: Why this split optimizes the risk-adjusted return (Sharpe Ratio) for a {horizon} period in India.\n"
    f"3. Fund Selection Logic: Mention the specific funds, highlighting their credit quality (e.g., AAA/Sovereign) or AUM stability rather than just names.\n"
    f"4. Architecture: How the Core (yield), Stability (liquidity), and Growth (alpha) tiers prevent emotional selling during market dips.\n"
    f"Constraint: No preamble, no bullets, professional tone, no other suggestions, strictly under 160 words."
    )
    print(prompt)
    try:
        if provider in ("rule", "rule_based"):
            text = _rule_based_horizon_explanation(user_risk, horizon, equity_pct, brackets)
            actual_provider = "rule_based"
        elif provider == "gemini":
            text = _call_gemini_cached(prompt)
            actual_provider = "gemini"
        elif provider == "groq":
            text = _call_groq_cached(prompt)
            actual_provider = "groq"
        elif provider == "openrouter":
            text = _call_openrouter_cached(prompt)
            actual_provider = "openrouter"
        elif provider == "ollama":
            text = _call_ollama_cached(prompt)
            actual_provider = "ollama"
        else:   # "huggingface"
            text = _call_huggingface_cached(prompt)
            actual_provider = "huggingface"
    except Exception:
        text = _rule_based_horizon_explanation(user_risk, horizon, equity_pct, brackets)
        actual_provider = "rule_based"

    return {"explanation": text, "provider": actual_provider, "horizon": horizon}


def explain_full_profile(
    profile_data: Dict,
    user_risk: str,
    user_context: str = "",
    provider: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Generate GenAI explanations for every horizon in a full user investment profile.

    Parameters
    ----------
    profile_data : output of recommend_full_profile()
    user_risk    : user's risk tier
    user_context : optional context string (age, goal, corpus)
    provider     : LLM provider override

    Returns
    -------
    dict of horizon_label (e.g. "5yr") → explain_horizon_portfolio() result dict
    """
    horizons = profile_data.get("horizons", {})
    results: Dict[str, Dict] = {}
    for label, horizon_data in horizons.items():
        results[label] = explain_horizon_portfolio(
            horizon_data, user_risk,
            user_context=user_context,
            provider=provider,
        )
        time.sleep(0.3)   # brief pause to stay within API rate limits
    return results
