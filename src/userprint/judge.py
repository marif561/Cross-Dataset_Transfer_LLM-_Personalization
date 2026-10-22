"""Step 5: LLM-as-a-judge pairwise comparison (paper Sect. 6).

For each dialogue and condition, every pair of models is shown to a judge together with the
persona summary and chat history; the judge answers A, B or TIE. The order of the two
responses is randomised to reduce position bias. Results: a win matrix per judge and a
win / loss / tie count per model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .llm import make_client
from .select import read_jsonl

JUDGE_PROMPT = """You are an impartial judge of personalised dialogue responses.

User persona: {persona}

Conversation so far:
{history}

Response A: {a}

Response B: {b}

Which response better fits both the conversation and the user persona? Answer with exactly one word: A, B, or TIE."""

VERDICT_RE = re.compile(r"\b(TIE|A|B)\b")


def parse_verdict(text: str) -> str:
    text = text.split("</think>")[-1].upper()
    m = VERDICT_RE.search(text)
    return m.group(1) if m else "TIE"


def judge_pairs(rows: List[Dict], dialogues: Dict, personas: Dict, generate, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    by_key: Dict = {}
    for r in rows:
        by_key.setdefault((r["id"], r["condition"]), {})[r["model"]] = r
    records = []
    for (did, cond), resp in by_key.items():
        d = dialogues[did]
        for m1, m2 in itertools.combinations(sorted(resp), 2):
            swap = rng.random() < 0.5
            a, b = (m2, m1) if swap else (m1, m2)
            prompt = JUDGE_PROMPT.format(persona=personas[resp[m1]["persona_id"]]["summary"],
                                         history="\n".join(d["history"]),
                                         a=resp[a]["response"], b=resp[b]["response"])
            v = parse_verdict(generate([{"role": "user", "content": prompt}]))
            winner = None if v == "TIE" else (a if v == "A" else b)
            records.append({"id": did, "condition": cond, "model_1": m1, "model_2": m2, "winner": winner})
    return pd.DataFrame(records)


def win_matrix(df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(set(df["model_1"]) | set(df["model_2"]))
    mat = pd.DataFrame(0, index=models, columns=models)
    for r in df.itertuples():
        if r.winner:
            loser = r.model_2 if r.winner == r.model_1 else r.model_1
            mat.loc[r.winner, loser] += 1
    return mat


def wlt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in sorted(set(df["model_1"]) | set(df["model_2"])):
        games = df[(df.model_1 == m) | (df.model_2 == m)]
        wins = int((games.winner == m).sum())
        ties = int(games.winner.isna().sum())
        rows.append({"model": m, "wins": wins, "losses": len(games) - wins - ties, "ties": ties})
    return pd.DataFrame(rows).set_index("model").sort_values("wins", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--judges", nargs="+", required=True, help="e.g. ollama:deepseek-r1 ollama:qwen3 ollama:gemma3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rows = read_jsonl(args.responses)
    dialogues = {d["id"]: d for d in read_jsonl(args.dialogues)}
    personas = {p["id"]: p for p in json.loads(Path(args.personas).read_text())["personas"]}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.judges:
        name = spec.replace(":", "_").replace("/", "_")
        df = judge_pairs(rows, dialogues, personas, make_client(spec, temperature=0.0, max_tokens=512), args.seed)
        df.to_csv(out / f"judgements_{name}.csv", index=False)
        win_matrix(df).to_csv(out / f"win_matrix_{name}.csv")
        for cond, part in df.groupby("condition"):
            print(f"[{spec}] {cond}: leaders {', '.join(wlt(part).index[:2])}")
        print(wlt(df).to_string())


if __name__ == "__main__":
    main()
