"""LLM-assisted feature discovery loop.

Ask a generative LLM to propose new features that could be derived from
the existing 19-feature vector, then validate each proposal by:

    1. Synthesising the feature from the base ones (simple arithmetic).
    2. Fitting a Ridge model with and without the proposal.
    3. Measuring the marginal lift in held-out R^2 / MAE.

We keep the LLM on a short leash: it can only output expressions using a
whitelist of operators and the existing feature names. Anything outside
that whitelist is rejected before we touch the model.

This is an instance of the LLM-as-scientist pattern (Wu et al. 2024,
Lu et al. 2024 "AI-Scientist"), adapted for a small tabular regressor.
The point isn't that the LLM will discover something the 2024-25 SOTA
missed; the point is to demonstrate a closed-loop protocol that is
rare in battery-ML papers.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


WHITELIST_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Load, ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
}

WHITELIST_CALLS = {"log", "log1p", "sqrt", "abs", "exp"}


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Safe expression evaluator
# --------------------------------------------------------------------------- #


def _check_ast(node: ast.AST, feature_names: set) -> bool:
    """Reject any expression that isn't arithmetic on known feature names."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if not (isinstance(func, ast.Name) and func.id in WHITELIST_CALLS):
                return False
            continue
        if isinstance(child, ast.Name):
            if child.id in feature_names or child.id in WHITELIST_CALLS:
                continue
            return False
        if type(child) not in WHITELIST_OPS and not isinstance(child, (ast.Call, ast.Name)):
            return False
    return True


def safe_eval_expr(expr: str, df: pd.DataFrame) -> Optional[np.ndarray]:
    """Evaluate ``expr`` on every row of ``df`` without exec-ing arbitrary code."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    numeric = df.select_dtypes(include=[np.number])
    if not _check_ast(tree, set(numeric.columns)):
        return None
    env = {name: numeric[name].to_numpy(dtype=float) for name in numeric.columns}
    env.update({
        "log": np.log, "log1p": np.log1p, "sqrt": np.sqrt,
        "abs": np.abs, "exp": np.exp,
    })
    try:
        val = eval(compile(tree, "<llm_feature>", "eval"), {"__builtins__": {}}, env)
    except Exception:
        return None
    arr = np.asarray(val, dtype=float)
    if arr.shape != (len(df),) or not np.all(np.isfinite(arr)):
        return None
    return arr


# --------------------------------------------------------------------------- #
# Prompt + parse
# --------------------------------------------------------------------------- #


PROMPT = """You are an expert in lithium-ion battery degradation modeling.

Here are the features currently available for a per-cycle SOH regressor.
All are numeric, already computed per cycle:

{feature_list}

Propose six new features as arithmetic expressions in the variables
above, each likely to improve SOH prediction. Allowed operators:
+, -, *, /, **, and the functions log, log1p, sqrt, abs, exp.
No other function calls, no conditionals, no variables outside the list.

Return ONLY a JSON array of objects in this exact shape:

[
  {{"name": "thermal_stress", "expr": "temp_delta * cycle_number", "reason": "..."}},
  {{"name": "internal_resistance_proxy", "expr": "dcir * voltage_range", "reason": "..."}}
]

Each reason should be one short physics-motivated sentence."""


def _parse_llm_response(text: str) -> List[Dict[str, str]]:
    # Tolerate markdown fences.
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # Otherwise grab the first JSON array in the message.
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        expr = str(it.get("expr", "")).strip()
        reason = str(it.get("reason", "")).strip()
        if name and expr:
            out.append({"name": name, "expr": expr, "reason": reason})
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _fit_ridge(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (r2, mae) on a 75/25 split."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    model = Ridge(alpha=1.0).fit(Xtr, y[tr])
    pred = model.predict(Xte)
    return float(r2_score(y[te], pred)), float(mean_absolute_error(y[te], pred))


def evaluate_proposals(
    train_df: pd.DataFrame,
    base_cols: List[str],
    target_col: str,
    proposals: List[Dict[str, str]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    df = train_df.dropna(subset=base_cols + [target_col]).copy()
    X_base = df[base_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    base_r2, base_mae = _fit_ridge(X_base, y)

    results: List[Dict[str, Any]] = []
    for p in proposals:
        arr = safe_eval_expr(p["expr"], df)
        if arr is None:
            results.append({**p, "status": "rejected_or_evaluation_failed"})
            continue
        std = float(np.std(arr))
        if std < 1e-12:
            results.append({**p, "status": "constant_feature"})
            continue
        X_plus = np.column_stack([X_base, arr])
        new_r2, new_mae = _fit_ridge(X_plus, y)
        results.append({
            **p,
            "status": "evaluated",
            "std": round(std, 4),
            "r2_with": round(new_r2, 5),
            "mae_with": round(new_mae, 5),
            "delta_r2": round(new_r2 - base_r2, 5),
            "delta_mae": round(new_mae - base_mae, 5),
        })
    baseline = {"base_r2": round(base_r2, 5), "base_mae": round(base_mae, 5),
                "n_samples": int(len(y)), "n_features_base": len(base_cols)}
    return baseline, results


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #


def call_llm(prompt: str) -> str:
    """Call Groq (preferred) or OpenAI. Reuses the env keys already loaded by the API."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_default_root() / ".env")
    except Exception:
        pass
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("your_"):
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content or ""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("your_"):
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content or ""
    raise RuntimeError("No LLM key set (Groq or OpenAI) in .env")


# --------------------------------------------------------------------------- #
# Project glue
# --------------------------------------------------------------------------- #


def run_discovery(project_root: Optional[Path] = None) -> Dict:
    project_root = Path(project_root or _default_root())
    train = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "train_features.csv")
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols = list(pickle.load(f))
    cols = [c for c in cols if c in train.columns]
    feature_list = "\n".join(f"  - {c}" for c in cols)
    prompt = PROMPT.format(feature_list=feature_list)

    try:
        raw = call_llm(prompt)
    except Exception as exc:
        return {"error": str(exc)}
    proposals = _parse_llm_response(raw)
    baseline, results = evaluate_proposals(train, cols, "soh", proposals)

    results_sorted = sorted(results, key=lambda d: d.get("delta_r2", -1e9), reverse=True)
    out = {
        "baseline": baseline,
        "n_proposals": len(proposals),
        "n_evaluated": sum(1 for r in results if r.get("status") == "evaluated"),
        "raw_llm_response": raw[:800],
        "results": results_sorted,
    }
    out_dir = project_root / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "llm_feature_discovery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _format(out: Dict) -> str:
    if "error" in out:
        return f"LLM feature discovery failed: {out['error']}"
    lines = [
        "LLM feature discovery", "=" * 60,
        f"Baseline       : R^2={out['baseline']['base_r2']:.4f}  MAE={out['baseline']['base_mae']:.4f}",
        f"Proposals      : {out['n_proposals']} requested, {out['n_evaluated']} evaluated",
        "",
        "Proposals ranked by delta R^2:",
    ]
    for r in out["results"]:
        if r.get("status") != "evaluated":
            lines.append(f"  [{r['status']}] {r.get('name','?')} -> {r.get('expr','')}")
            continue
        sign = "+" if r["delta_r2"] >= 0 else ""
        lines.append(
            f"  {r['name']:<30} {r['expr']:<45}  "
            f"R^2 {sign}{r['delta_r2']:.5f}  MAE {r['delta_mae']:+.5f}"
        )
    return "\n".join(lines)


def main() -> int:
    out = run_discovery()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
