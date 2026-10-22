"""Step 3: generate responses for PersonaChat dialogues under the three conditions.

Dialogues are JSONL rows ``{"id", "history": [turn, ...], "reference": "gold next turn"}``.
Each dialogue is matched to the USERPRINT persona whose centroid is closest to its
history embedding (cross-dataset transfer: personas come from LMSYS, dialogues from
PersonaChat). Output rows: ``{"id", "model", "condition", "persona_id", "response", "reference"}``.
Re-running the command skips rows that already exist, so long runs can be resumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import CONDITIONS
from .embed import Embedder
from .llm import make_client
from .prompts import build_messages
from .select import read_jsonl


def assign_personas(dialogues: List[Dict], personas: List[Dict], embedder: Embedder) -> List[int]:
    cents = np.array([p["centroid"] for p in personas], dtype="float32")
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    emb = embedder.encode([" ".join(d["history"]) for d in dialogues])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return (emb @ cents.T).argmax(1).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, help="PersonaChat-style JSONL")
    ap.add_argument("--personas", required=True, help="personas JSON from userprint.personas")
    ap.add_argument("--models", nargs="+", required=True, help="e.g. ollama:llama4 ollama:gemma3:27b openai:gpt-4")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=300, help="number of dialogues (300 in the paper)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    meta = json.loads(Path(args.personas).read_text(encoding="utf-8"))
    personas = meta["personas"]
    if not personas:
        raise SystemExit("no personas in file — lower --min-cluster-size when clustering")
    dialogues = read_jsonl(args.dialogues)[: args.limit]
    pid = assign_personas(dialogues, personas, Embedder(meta["embedder"]))

    out = Path(args.output)
    done = {(r["id"], r["model"], r["condition"]) for r in read_jsonl(out)} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for spec in args.models:
            gen = make_client(spec)
            for d, p in zip(dialogues, pid):
                for cond in args.conditions:
                    if (d["id"], spec, cond) in done:
                        continue
                    resp = gen(build_messages(cond, d["history"], personas[p]))
                    fh.write(json.dumps({"id": d["id"], "model": spec, "condition": cond,
                                         "persona_id": personas[p]["id"], "response": resp,
                                         "reference": d["reference"]}, ensure_ascii=False) + "\n")
                    fh.flush()
            print(f"{spec}: done")


if __name__ == "__main__":
    main()
