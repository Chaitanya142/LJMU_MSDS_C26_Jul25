"""
test_llm_providers.py
=====================
End-to-end tests for the free LLM provider chain in smart_fund_advisor.

Tests:
  1. Provider detection  — which providers are configured
  2. Rule-based baseline — always works, no API key needed
  3. Live LLM call       — uses whichever provider is configured (skips if none)
  4. Gemini direct test  — if GEMINI_API_KEY is set
  5. Groq direct test    — if GROQ_API_KEY is set
  6. HuggingFace test    — if HF_TOKEN is set
  7. Ollama test         — if Ollama is running locally
  8. Validation check    — correctness scoring on LLM output
  9. Portfolio batch     — multi-fund explanation + batch validate

Run with:
    conda activate smart_fund_advisor
    python test_llm_providers.py

To test a specific provider:
    GEMINI_API_KEY=<key>  python test_llm_providers.py
    GROQ_API_KEY=<key>    python test_llm_providers.py
    HF_TOKEN=<token>      python test_llm_providers.py

Free API keys:
  Gemini:     https://aistudio.google.com/apikey   (1500 req/day, no credit card)
  Groq:       https://console.groq.com             (6000 tokens/min free)
  OpenRouter: https://openrouter.ai/keys           (free tier models)
  HuggingFace:https://huggingface.co/settings/tokens (free read token)
  Ollama:     brew install ollama && ollama pull llama3.2 && ollama serve
"""

import os
import sys
import time
import textwrap
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ── Colours for terminal output ────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{'─'*60}\n  {title}\n{'─'*60}{RESET}")


# ── Sample fund row (Mirae Asset Large Cap — representative High-risk fund) ───
SAMPLE_FUND = pd.Series({
    "Scheme_Name":        "Mirae Asset Large Cap Fund",
    "Scheme_Category":    "Large Cap Fund",
    "AMC":                "Mirae Asset",
    "NAV":                94.5,
    "Average_AUM_Cr":     35000,
    "cagr_3yr_hist":      0.148,
    "sharpe_1yr_hist":    1.12,
    "max_drawdown_hist":  -0.18,
    "expense_ratio_real": 0.55,
    "ter_missing_flag":   0,
    "cluster_label":      "Growth Seeker",
})

USER_RISK    = "High"
USER_CONTEXT = "30-year-old salaried professional, 5-year investment horizon, ₹50k monthly SIP budget"

_PASS_COUNT  = 0
_FAIL_COUNT  = 0


def record(passed: bool):
    global _PASS_COUNT, _FAIL_COUNT
    if passed:
        _PASS_COUNT += 1
    else:
        _FAIL_COUNT += 1


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Provider detection
# ══════════════════════════════════════════════════════════════════════════════

section("TEST 1 — Provider Detection")

from src.gpt_explainer import (
    _GENAI_AVAILABLE, _GROQ_SDK_AVAILABLE,
    _check_ollama_running, get_active_provider,
)

keys = {
    "GEMINI_API_KEY":     os.getenv("GEMINI_API_KEY",    ""),
    "GROQ_API_KEY":       os.getenv("GROQ_API_KEY",      ""),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY",""),
    "HF_TOKEN":           os.getenv("HF_TOKEN",          ""),
}

for name, val in keys.items():
    if val:
        ok(f"{name} = {'*' * min(len(val)-4,8)}{val[-4:]}  (set)")
    else:
        warn(f"{name} not set  →  provider unavailable")

ollama_up = _check_ollama_running()
if ollama_up:
    ok("Ollama is running at localhost:11434")
else:
    warn("Ollama not running  →  install: brew install ollama && ollama serve")

active = get_active_provider()
info(f"google-genai SDK installed : {_GENAI_AVAILABLE}")
info(f"groq SDK installed         : {_GROQ_SDK_AVAILABLE}")
info(f"Active provider selected   : {BOLD}{active}{RESET}")

has_real_llm = any(keys.values()) or ollama_up
if has_real_llm:
    ok(f"At least one real LLM provider is available: {active}")
    record(True)
else:
    warn("No API keys set and Ollama is not running.")
    warn("Tests will fall back to rule-based. Set any free key to test real LLM.")
    warn("Fastest option: GEMINI_API_KEY — get free at https://aistudio.google.com/apikey")
    record(False)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Rule-based baseline (always works)
# ══════════════════════════════════════════════════════════════════════════════

section("TEST 2 — Rule-Based Baseline (no API key needed)")

from src.gpt_explainer import explain_fund, validate_gpt_correctness

text, prov = explain_fund(SAMPLE_FUND, USER_RISK, USER_CONTEXT, provider="rule")
assert prov == "rule_based", f"Expected rule_based, got {prov}"
assert "Mirae Asset" in text, "Fund name missing from rule-based output"
assert len(text) > 80, f"Rule-based output too short: {len(text)} chars"

ok(f"Provider used  : {prov}")
ok(f"Output length  : {len(text)} characters")
v = validate_gpt_correctness(text, SAMPLE_FUND, USER_RISK)
ok(f"Correctness    : {v['correctness_score']:.2f}  ({v['verdict']})")
print(f"\n  Preview:\n{textwrap.indent(text[:300], '    ')}")
record(v["pass"])


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Auto-detected provider (uses best available key)
# ══════════════════════════════════════════════════════════════════════════════

section(f"TEST 3 — Auto-detected Provider  [{active}]")

t0 = time.time()
text3, prov3 = explain_fund(
    SAMPLE_FUND, USER_RISK, USER_CONTEXT,
    provider=None,   # auto-detect
    fallback=True,
)
elapsed = time.time() - t0

ok(f"Provider used  : {BOLD}{prov3}{RESET}")
ok(f"Response time  : {elapsed:.1f}s")
ok(f"Output length  : {len(text3)} characters")

v3 = validate_gpt_correctness(text3, SAMPLE_FUND, USER_RISK)
ok(f"Correctness    : {v3['correctness_score']:.2f}  ({v3['verdict']})")

is_real_llm = prov3 not in ("rule_based",)
if is_real_llm:
    ok(f"REAL LLM response obtained via '{prov3}'!")
    print(f"\n  Full LLM Response:\n  {'─'*50}")
    print(textwrap.indent(text3.strip(), "  "))
    print(f"  {'─'*50}")
else:
    warn(f"Fell back to rule_based (no API keys set) — set GEMINI_API_KEY for real LLM")
    info(f"  Preview: {text3[:200]!r}")

record(v3["pass"])
record(is_real_llm)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4-7: Individual provider tests (skip if key/service not available)
# ══════════════════════════════════════════════════════════════════════════════

def _test_provider(name: str, call_fn, display_name: str):
    section(f"TEST — {display_name} Direct Call")
    try:
        t0 = time.time()
        raw = call_fn()
        elapsed = time.time() - t0
        ok(f"Response received in {elapsed:.1f}s ({len(raw)} chars)")
        ok(f"Provider: {name}")
        v = validate_gpt_correctness(raw, SAMPLE_FUND, USER_RISK)
        ok(f"Correctness: {v['correctness_score']:.2f}  ({v['verdict']})")
        print(f"\n  LLM Response:\n  {'─'*50}")
        print(textwrap.indent(raw[:600].strip(), "  "))
        print(f"  {'─'*50}")
        record(v["pass"])
        return True
    except Exception as exc:
        fail(f"{display_name} call failed: {exc}")
        record(False)
        return False


from src.gpt_explainer import (
    _call_gemini, _call_groq, _call_huggingface, _call_ollama,
)

# Build a minimal prompt for direct provider tests
_PROMPT = (
    "Explain in 3 short paragraphs why the mutual fund "
    "'Mirae Asset Large Cap Fund' by Mirae Asset "
    "(Category: Large Cap Fund, NAV: ₹94.50, AUM: ₹35,000 Cr, "
    "3yr CAGR: 14.8%, Sharpe: 1.12, TER: 0.55%) "
    "is suitable for a High risk investor. "
    "User profile: 30-year-old salaried, 5yr horizon. "
    "Under 180 words. Be specific to Indian market."
)

if keys["GEMINI_API_KEY"]:
    _test_provider("gemini", lambda: _call_gemini(_PROMPT, 400), "Google Gemini 2.0 Flash")
else:
    warn("TEST 4 — Gemini: GEMINI_API_KEY not set → SKIPPED")
    warn("  Get free key: https://aistudio.google.com/apikey")

if keys["GROQ_API_KEY"]:
    _test_provider("groq", lambda: _call_groq(_PROMPT, 400), "Groq Llama-3.3-70B")
else:
    warn("TEST 5 — Groq: GROQ_API_KEY not set → SKIPPED")
    warn("  Get free key: https://console.groq.com")

if keys["HF_TOKEN"]:
    _test_provider("huggingface", lambda: _call_huggingface(_PROMPT, 300), "HuggingFace Qwen2.5-72B")
else:
    warn("TEST 6 — HuggingFace: HF_TOKEN not set → SKIPPED")
    warn("  Get free token: https://huggingface.co/settings/tokens")

if ollama_up:
    _test_provider("ollama", lambda: _call_ollama(_PROMPT, 400), "Ollama (local model)")
else:
    warn("TEST 7 — Ollama: server not running → SKIPPED")
    warn("  macOS install: brew install ollama && ollama pull llama3.2 && ollama serve")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Validation edge cases
# ══════════════════════════════════════════════════════════════════════════════

section("TEST 8 — Validation Edge Cases")

from src.gpt_explainer import validate_gpt_correctness

# Should PASS: high quality text with all required elements
good_text = (
    "Mirae Asset Large Cap Fund is a well-diversified large-cap equity fund with "
    "NAV ₹94.50 and AUM ₹35,000 Cr. The 3-year CAGR of 14.8% and Sharpe ratio of "
    "1.12 demonstrate consistent risk-adjusted returns. For a High risk investor "
    "seeking long-term wealth creation, this large cap fund offers exposure to "
    "blue-chip Indian equities with controlled volatility. The low TER of 0.55% "
    "maximises net returns over a 5+ year horizon."
)
vg = validate_gpt_correctness(good_text, SAMPLE_FUND, USER_RISK)
assert vg["pass"], f"Good text should PASS; got score {vg['correctness_score']}"
ok(f"Good text  → score {vg['correctness_score']:.2f}  PASS  ✓")
record(True)

# Should FAIL: hallucinates NAV (very wrong value)
bad_text = (
    "This fund has a NAV of ₹9450 (10× actual) and is a medium risk scheme "
    "suitable for conservative investors."
)
vb = validate_gpt_correctness(bad_text, SAMPLE_FUND, USER_RISK)
assert not vb["pass"], f"Bad text (hallucinated NAV) should FAIL; got score {vb['correctness_score']}"
ok(f"Bad text   → score {vb['correctness_score']:.2f}  FAIL  ✓  (correctly detected hallucination)")
record(True)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Portfolio batch explanation
# ══════════════════════════════════════════════════════════════════════════════

section("TEST 9 — Portfolio Batch Explanation")

from src.gpt_explainer import explain_portfolio, batch_validate

# Build a small 3-row dataframe of funds
funds_df = pd.DataFrame([
    {
        "Scheme_Name": "Mirae Asset Large Cap Fund", "Scheme_Category": "Large Cap Fund",
        "AMC": "Mirae Asset", "NAV": 94.5, "Average_AUM_Cr": 35000,
        "cagr_3yr_hist": 0.148, "sharpe_1yr_hist": 1.12, "expense_ratio_real": 0.55,
        "ter_missing_flag": 0,
    },
    {
        "Scheme_Name": "ICICI Pru Balanced Advantage", "Scheme_Category": "Dynamic Asset Allocation",
        "AMC": "ICICI Prudential", "NAV": 58.2, "Average_AUM_Cr": 52000,
        "cagr_3yr_hist": 0.123, "sharpe_1yr_hist": 0.91, "expense_ratio_real": 1.05,
        "ter_missing_flag": 0,
    },
    {
        "Scheme_Name": "SBI Small Cap Fund", "Scheme_Category": "Small Cap Fund",
        "AMC": "SBI Mutual Fund", "NAV": 142.0, "Average_AUM_Cr": 22000,
        "cagr_3yr_hist": 0.281, "sharpe_1yr_hist": 1.43, "expense_ratio_real": 1.50,
        "ter_missing_flag": 0,
    },
])

t0 = time.time()
explanations = explain_portfolio(funds_df, USER_RISK, USER_CONTEXT, top_n=3, provider=None)
elapsed = time.time() - t0

ok(f"Generated {len(explanations)} fund explanations in {elapsed:.1f}s")

providers_used = {v["provider"] for v in explanations.values()}
info(f"Providers used: {providers_used}")

# Any real LLM call is a win
real_llm_used = any(p not in ("rule_based", "unknown") for p in providers_used)
if real_llm_used:
    ok(f"Real LLM provider(s) used: {providers_used - {'rule_based','unknown'}}")

batch = batch_validate(funds_df, explanations, USER_RISK)
ok(f"Batch validation: {batch['pass_count']}/{batch['total']} funds passed")
ok(f"Overall pass rate: {batch['overall_pass_rate']:.2f}")
record(batch["overall_pass"])

# Print one sample explanation
if explanations:
    fname, fdata = next(iter(explanations.items()))
    print(f"\n  Sample explanation for '{fname}' [{fdata['provider']}]:")
    print(f"  {'─'*50}")
    print(textwrap.indent(fdata["explanation"][:500].strip(), "  "))
    print(f"  {'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")

total = _PASS_COUNT + _FAIL_COUNT
print(f"\n  Passed : {GREEN}{_PASS_COUNT}{RESET} / {total}")
print(f"  Failed : {RED}{_FAIL_COUNT}{RESET} / {total}")

if _FAIL_COUNT == 0:
    print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED ✓{RESET}")
elif _FAIL_COUNT <= 2 and not has_real_llm:
    print(f"\n  {YELLOW}{BOLD}Tests passed (rule-based only). Set a free API key for live LLM.{RESET}")
    print()
    print("  NEXT STEP — get any one free key:")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Gemini (easiest, 1500 req/day, no credit card):            │")
    print("  │    export GEMINI_API_KEY=<key>                              │")
    print("  │    → https://aistudio.google.com/apikey                    │")
    print("  │                                                             │")
    print("  │  Groq (llama-3.3-70B, 6000 tok/min):                       │")
    print("  │    export GROQ_API_KEY=<key>                                │")
    print("  │    → https://console.groq.com                              │")
    print("  │                                                             │")
    print("  │  Ollama (100% local, no internet needed):                   │")
    print("  │    brew install ollama && ollama pull llama3.2              │")
    print("  │    ollama serve   (in a separate terminal)                  │")
    print("  └─────────────────────────────────────────────────────────────┘")
else:
    print(f"\n  {RED}{BOLD}Some tests FAILED — check output above.{RESET}")

print()
