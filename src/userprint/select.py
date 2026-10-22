"""Step 1: extract user utterances and keep the most diverse, information-seeking ones.

The paper keeps the top quartile of LMSYS Chatbot Arena user utterances by lexical
variation and information-seeking intent. Here:
  * lexical variation = moving-average type-token ratio (MATTR, window 25) — robust to length
  * information-seeking = share of question / request cues (``?``, wh-words, "how to", "explain", ...)
Both are min-max scaled and averaged; utterances at or above the 75th percentile are kept.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z']+")
INFO_CUES = re.compile(
    r"\?|\b(what|why|how|when|where|which|who|whom|whose|can you|could you|would you|is there|are there|"
    r"explain|describe|tell me|help me|show me|compare|difference|recommend|suggest|list|give me)\b",
    re.IGNORECASE)


def user_turns(record: Dict) -> List[str]:
    """User utterances from an LMSYS-style record.

    Accepts ``conversation`` / ``conversation_a`` (list of {role, content}) or a plain ``text`` field.
    """
    for key in ("conversation", "conversation_a", "messages"):
        if isinstance(record.get(key), list):
            return [t.get("content", "") for t in record[key] if t.get("role") in ("user", "human")]
    return [record["text"]] if record.get("text") else []


def mattr(tokens: List[str], window: int = 25) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


def info_seeking(text: str) -> float:
    n_sent = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    return min(1.0, len(INFO_CUES.findall(text)) / n_sent)


def score(texts: Iterable[str], min_tokens: int = 5) -> List[Dict]:
    rows = []
    for t in texts:
        toks = [w.lower() for w in TOKEN_RE.findall(t)]
        if len(toks) >= min_tokens:
            rows.append({"text": t.strip(), "lexical": mattr(toks), "info": info_seeking(t), "n_tokens": len(toks)})
    if not rows:
        return rows

    def scale(k):
        v = np.array([r[k] for r in rows], dtype=float)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    combined = (scale("lexical") + scale("info")) / 2
    for r, c in zip(rows, combined):
        r["score"] = float(c)
    return rows


def top_quartile(rows: List[Dict], q: float = 0.75) -> List[Dict]:
    if not rows:
        return []
    cut = np.quantile([r["score"] for r in rows], q)
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: -r["score"]):
        key = r["text"].lower()
        if r["score"] >= cut and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def read_jsonl(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL of conversations (LMSYS Chatbot Arena format)")
    ap.add_argument("--output", required=True, help="JSONL of selected utterances")
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--min-tokens", type=int, default=5)
    args = ap.parse_args(argv)

    texts = [t for rec in read_jsonl(args.input) for t in user_turns(rec)]
    kept = top_quartile(score(texts, args.min_tokens), args.quantile)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(texts):,} user utterances -> {len(kept):,} kept (top {1 - args.quantile:.0%})")


if __name__ == "__main__":
    main()
