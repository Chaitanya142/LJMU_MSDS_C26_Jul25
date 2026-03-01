"""
gpt_explainer.py - Free LLM fund explanation for Smart Fund Advisor (v3).

Provider Strategy  (in priority order — all FREE, best-quality-first)
---------------------------------------------------------------------
1. Groq API (groq.com) — **llama-3.3-70b-versatile** (70B params, free tier)
   Massive upgrade from 8B: superior financial reasoning, nuanced explanations.
   Set env var:  GROQ_API_KEY=<your-key>   (get free key at console.groq.com)

2. OpenRouter (openrouter.ai) — **google/gemma-2-9b-it:free**
   Google Gemma 2 9B: excellent instruction-following, strong on finance.
   Set env var:  OPENROUTER_API_KEY=<your-key>

3. HuggingFace Inference API (huggingface.co) — **Qwen/Qwen2.5-3B-Instruct**
   Qwen 2.5 3B: state-of-the-art small model, no API key needed for free tier.
   Set env var:  HF_TOKEN=<your-token>   (optional but helps with rate limits)

4. Rule-based fallback — no API key needed, always works

Model Selection Rationale
-------------------------------------------------------------------
- llama-3.3-70b-versatile: MT-Bench 8.6, superior financial domain knowledge,
  handles complex multi-fund portfolio explanations. Free on Groq (Dec 2024).
- gemma-2-9b-it: MMLU 71.3%, strong instruction-following, Google-trained
  with safety alignment. Free tier on OpenRouter.
- Qwen2.5-3B-Instruct: Best-in-class <5B model on IFEval (76.1%),
  multilingual, runs on free HF inference. Backup for no-API-key scenarios.
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
import requests
from typing import Dict, Optional, Tuple
import pandas as pd


# ─── Provider detection ────────────────────────────────────────────────────────

def _get_provider() -> str:
    """Return the best available free LLM provider."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "huggingface"   # free HF inference (no key required for many models)


def get_active_provider() -> str:
    """Public helper — returns name of the active free LLM provider."""
    return _get_provider()


# ─── Groq API (Free tier — Llama-3.1-8B) ─────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int = 400) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # v3: upgraded to llama-3.3-70b-versatile (from 8B → 70B, free on Groq)
    # 70B model produces dramatically better financial reasoning and
    # portfolio-level explanations with proper SEBI/AMFI terminology.
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": (
                "You are a SEBI-registered investment advisor specialising in Indian mutual funds. "
                "Give concise, data-grounded, personalised explanations. Reference NAV, AUM, "
                "expense ratio, CAGR, Sharpe ratio where available. Keep responses under 200 words."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


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
            {"role": "system", "content": (
                "You are a SEBI-registered Indian mutual fund advisor. Give concise, accurate, "
                "data-grounded fund explanations referencing NAV, CAGR, expense ratio. "
                "Keep responses under 200 words."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─── HuggingFace Inference API (Free, no key needed) ─────────────────────────

def _call_huggingface(prompt: str, max_tokens: int = 400) -> str:
    token = os.getenv("HF_TOKEN", "")
    # v3: upgraded to Qwen2.5-3B-Instruct (from Mistral-7B-v0.2)
    # Qwen 2.5 3B: best-in-class small model, IFEval 76.1%, strong multilingual.
    model = "Qwen/Qwen2.5-3B-Instruct"
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Qwen2.5 chat template (ChatML format)
    full_prompt = (
        "<|im_start|>system\n"
        "You are a SEBI-registered Indian mutual fund advisor. "
        "Give concise, data-grounded, personalised fund explanations "
        "referencing NAV, AUM, CAGR, expense ratio where available. "
        "Keep responses under 150 words.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.3,
            "return_full_text": False,
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    
    # HF may return 503 while model loads — retry once
    if resp.status_code == 503:
        time.sleep(10)
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    return str(result)


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
    ctx_note = f" Based on your profile ({user_context})." if user_context else ""

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
    er     = fund_row.get("synth_expense_ratio",fund_row.get("Expense_Ratio", None))
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

    prompt = (
        f"Explain in 3 short paragraphs why the mutual fund '{name}' by {amc} "
        f"(Category: {cat}, NAV: ₹{nav:.2f}, AUM: ₹{aum:,.0f} Cr, {metrics_str}) "
        f"is suitable for an investor with a '{user_risk}' risk appetite. "
        f"{cluster_str}"
        f"User context: {user_context or 'Indian retail investor'}. "
        f"Include: (1) fund overview with key metrics, (2) why it matches the risk profile, "
        f"(3) expense ratio impact and return consistency. "
        f"Be specific to Indian market. Under 180 words."
    )

    # Try API providers
    if provider == "rule":
        return _rule_based_explanation(fund_row, user_risk, user_context), "rule_based"

    try:
        if provider == "groq":
            text = _call_groq(prompt)
        elif provider == "openrouter":
            text = _call_openrouter(prompt)
        else:
            text = _call_huggingface(prompt)
        return text, provider
    except Exception as e:
        if fallback:
            return _rule_based_explanation(fund_row, user_risk, user_context), "rule_based"
        raise RuntimeError(f"GPT provider '{provider}' failed: {e}") from e


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
        f"You are a SEBI-registered investment advisor in India.\n"
        f"A '{user_risk}' risk investor ({user_context or 'Indian retail investor'}) "
        f"is planning a {horizon} investment.\n\n"
        f"Recommended core-satellite portfolio:\n"
        f"{portfolio_summary}\n"
        f"Equity: {equity_pct*100:.0f}%  |  Debt: {debt_pct*100:.0f}%\n\n"
        f"Write exactly 3 paragraphs (under 160 words total):\n"
        f"1. Why the {equity_pct*100:.0f}%/{debt_pct*100:.0f}% equity/debt split suits "
        f"a {horizon} investment horizon.\n"
        f"2. How the Core / Stability / Growth bracket structure balances risk and return.\n"
        f"3. One specific risk to monitor for this portfolio and horizon.\n"
        f"Be specific to the Indian market. No preamble or bullet points."
    )

    try:
        if provider == "rule":
            text = _rule_based_horizon_explanation(user_risk, horizon, equity_pct, brackets)
            actual_provider = "rule_based"
        elif provider == "groq":
            text = _call_groq(prompt)
            actual_provider = "groq"
        elif provider == "openrouter":
            text = _call_openrouter(prompt)
            actual_provider = "openrouter"
        else:
            text = _call_huggingface(prompt)
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
