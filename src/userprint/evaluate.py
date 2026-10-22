"""Step 4: automatic evaluation (paper Sect. 4).

Per response: ROUGE-L F1 (lexical overlap) and BERTScore F1 (semantic similarity) against
the PersonaChat reference. Per model:
  * mean score under each condition (Table 1)
  * gains  Δ1 = P − C,  Δ2 = P+C − C,  Δ3 = P+C − P  (Table 2)
  * standard deviation of the per-condition means (consistency)
  * bonus efficiency = (P+C − C) / (P+C − P): how much chat history adds on top of the persona
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .select import read_jsonl


def _lcs(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def bertscore(hyps: List[str], refs: List[str], model: str = "roberta-large") -> List[float]:
    from bert_score import score
    _, _, f1 = score(hyps, refs, model_type=model, lang="en", verbose=False)
    return f1.tolist()


def summarise(df: pd.DataFrame, metrics=("bertscore", "rougeL")) -> Dict[str, pd.DataFrame]:
    means = df.pivot_table(index="model", columns="condition", values=list(metrics), aggfunc="mean")
    rows = []
    for model in means.index:
        row = {"model": model}
        for m in metrics:
            c, p, pc = (means.loc[model, (m, k)] if (m, k) in means.columns else np.nan
                        for k in ("chat", "persona", "persona+chat"))
            row[f"{m}_d1_P-C"], row[f"{m}_d2_PC-C"], row[f"{m}_d3_PC-P"] = p - c, pc - c, pc - p
            row[f"{m}_std"] = float(np.nanstd([c, p, pc]))
            denom = pc - p
            row[f"{m}_bonus_efficiency"] = (pc - c) / denom if denom and not np.isnan(denom) else np.nan
        rows.append(row)
    return {"means": means, "gains": pd.DataFrame(rows).set_index("model")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="JSONL from userprint.generate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-bertscore", action="store_true", help="ROUGE-L only (no model download)")
    ap.add_argument("--bertscore-model", default="roberta-large")
    args = ap.parse_args(argv)

    df = pd.DataFrame(read_jsonl(args.responses))
    df["rougeL"] = [rouge_l(h, r) for h, r in zip(df["response"], df["reference"])]
    metrics = ["rougeL"]
    if not args.no_bertscore:
        df["bertscore"] = bertscore(df["response"].tolist(), df["reference"].tolist(), args.bertscore_model)
        metrics.insert(0, "bertscore")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores_per_response.csv", index=False)
    s = summarise(df, metrics)
    s["means"].round(4).to_csv(out / "means_by_condition.csv")
    s["gains"].round(4).to_csv(out / "gains_and_consistency.csv")
    if "bertscore" in df:
        print(f"BERTScore–ROUGE-L correlation r = {df['bertscore'].corr(df['rougeL']):.2f}")
    print(s["means"].round(3).to_string())
    print(s["gains"].round(4).to_string())


if __name__ == "__main__":
    main()
