#!/usr/bin/env python3
"""
unicode_math_verifier.py

Shared answer verifier for the unicode-format RLVR project.
Used by the DeepMath baseline evaluation now; designed to be lifted
into the training reward function later so that baseline eval,
training reward, and validation all judge answers IDENTICALLY.

Verification chain for a model response:
  1. extract the 《answer》...《/answer》 block (format discipline);
     if absent, fall back to the full response
  2. math-verify (LaTeX -> sympy symbolic equivalence) on the
     extracted text vs the gold answer            <- primary
  3. v3-style normalized string / numeric match    <- fallback

Requires:  pip install math-verify
"""

import re
from fractions import Fraction

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:  # degrade gracefully; caller can assert if required
    HAS_MATH_VERIFY = False

ANSWER_BLOCK = re.compile(r"《answer》\s*(.*?)\s*《/answer》", re.S)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def extract_answer_block(response: str):
    """Return (text, found_block). Last 《answer》 block wins."""
    m = ANSWER_BLOCK.findall(response or "")
    if m:
        return m[-1].strip(), True
    return (response or "").strip(), False


# ---------------------------------------------------------------------------
# fallback matcher (v3-compatible: unbox + normalize + numeric equivalence)
# ---------------------------------------------------------------------------

def _unbox(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        j = s.find("\\boxed{", i)
        if j == -1:
            out.append(s[i:]); break
        out.append(s[i:j])
        k, depth = j + len("\\boxed{"), 1
        start = k
        while k < n and depth > 0:
            if s[k] == "{": depth += 1
            elif s[k] == "}": depth -= 1
            k += 1
        out.append(s[start:k - 1] if depth == 0 else s[start:])
        i = k
    return "".join(out)


def _normalize(x) -> str:
    if x is None:
        return ""
    s = _unbox(str(x).strip())
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


_FRAC_TEX = re.compile(r"^\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}$")


def _to_number(norm: str):
    if not norm:
        return None
    s, pct = norm, False
    if s.endswith("%"):
        pct, s = True, s[:-1]
    try:
        m = _FRAC_TEX.match(s)
        val = Fraction(m.group(1)) / Fraction(m.group(2)) if m else Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return val / 100 if pct else val


def fallback_match(pred, gold) -> bool:
    a, b = _normalize(pred), _normalize(gold)
    if a and a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    return na is not None and nb is not None and na == nb


# ---------------------------------------------------------------------------
# primary: math-verify
# ---------------------------------------------------------------------------

def math_verify_match(pred_text: str, gold: str) -> bool:
    """Symbolic equivalence via math-verify. pred_text may be a full
    passage (math-verify extracts \\boxed{} / last math itself)."""
    if not HAS_MATH_VERIFY:
        return False
    try:
        g = mv_parse(f"${gold}$")
        p = mv_parse(pred_text)
        if mv_verify(g, p):
            return True
        # some responses state the bare answer without boxing/latex:
        p2 = mv_parse(f"${pred_text}$")
        return bool(mv_verify(g, p2))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def judge(response: str, gold: str) -> dict:
    """Full verdict for one model response against a gold answer."""
    block, has_block = extract_answer_block(response)
    mv = math_verify_match(block, gold)
    fb = False if mv else fallback_match(block, gold)
    return {
        "correct": bool(mv or fb),
        "via": "math_verify" if mv else ("fallback" if fb else "none"),
        "answer_block_found": has_block,
    }


if __name__ == "__main__":
    assert HAS_MATH_VERIFY, "pip install math-verify first"
    r = "《reasoning》 steps... 《/reasoning》 《answer》 \\boxed{\\frac{-1+\\sqrt{5}}{2}} 《/answer》"
    v = judge(r, r"\frac{\sqrt{5}-1}{2}")
    assert v["correct"] and v["via"] == "math_verify" and v["answer_block_found"], v
    v = judge("the answer is 1/2", "0.5")
    assert v["correct"], v
    v = judge("《answer》 35 《/answer》", "34")
    assert not v["correct"], v
    print("self-test OK")
