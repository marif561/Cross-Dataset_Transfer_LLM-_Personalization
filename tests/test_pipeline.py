import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from userprint.embed import Embedder
from userprint.evaluate import rouge_l, summarise
from userprint.evaluate import main as eval_main
from userprint.generate import main as gen_main
from userprint.judge import judge_pairs, parse_verdict, win_matrix, wlt
from userprint.personas import build_personas, cluster, keywords_per_cluster, main as personas_main
from userprint.prompts import build_messages
from userprint.select import info_seeking, mattr, main as select_main, score, top_quartile, user_turns

S = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_user_turns_lmsys_format():
    rec = {"conversation_a": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"},
                              {"role": "user", "content": "q2"}]}
    assert user_turns(rec) == ["q1", "q2"] and user_turns({"text": "x"}) == ["x"]


def test_selection_prefers_diverse_information_seeking():
    assert mattr("ok ok ok ok".split()) < mattr("how does garbage collection work".split())
    assert info_seeking("How do I fix this?") > info_seeking("I fixed it yesterday.")
    rows = score(["ok ok ok ok ok ok", "thanks thanks thanks thanks thanks",
                  "How do I tune garbage collection pauses in a Java service?",
                  "Can you explain why my Docker container cannot reach the database?"])
    kept = {r["text"] for r in top_quartile(rows)}
    assert all("ok ok" not in k and "thanks" not in k for k in kept) and kept


def test_clustering_recovers_separated_groups():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(c, 0.05, (20, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    labels = cluster(pts, min_cluster_size=5)
    assert len(set(labels) - {-1}) == 3


def test_keywords_and_persona_summary():
    texts = ["python code bug", "python test failing", "trip to japan", "japan travel visa"]
    labels = np.array([0, 0, 1, 1])
    kw = keywords_per_cluster(texts, labels)
    assert "python" in kw[0] and "japan" in kw[1]
    emb = Embedder("hash").encode(texts)
    personas = build_personas(texts, emb, labels, generate=lambda m: "A curious traveller.")
    assert personas[1]["summary"] == "A curious traveller." and personas[0]["size"] == 2


def test_three_conditions():
    p = {"summary": "A runner who likes data.", "keywords": ["running", "python"]}
    hist = ["hi", "hello, how are you?", "tired after my run"]
    chat = build_messages("chat", hist, None)
    pers = build_messages("persona", hist, p)
    both = build_messages("persona+chat", hist, p)
    assert "persona" not in chat[0]["content"].lower() and chat[-1] == {"role": "user", "content": "tired after my run"}
    assert len(pers) == 2 and "runner" in pers[0]["content"]
    assert len(both) == 4 and both[1]["role"] == "user" and both[2]["role"] == "assistant"
    with pytest.raises(ValueError):
        build_messages("persona", hist, None)


def test_rouge_l():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert rouge_l("", "x") == 0.0
    assert 0 < rouge_l("the cat is here", "the cat sat on the mat") < 1


def test_gains_and_bonus_efficiency():
    df = pd.DataFrame([{"model": "m", "condition": c, "rougeL": v}
                       for c, v in (("chat", 0.10), ("persona", 0.12), ("persona+chat", 0.16))])
    g = summarise(df, ["rougeL"])["gains"].loc["m"]
    assert g["rougeL_d1_P-C"] == pytest.approx(0.02)
    assert g["rougeL_d2_PC-C"] == pytest.approx(0.06)
    assert g["rougeL_bonus_efficiency"] == pytest.approx(0.06 / 0.04)


def test_judge_parsing_and_aggregation():
    assert parse_verdict("<think>hmm A or B</think> B") == "B"
    assert parse_verdict("Tie.") == "TIE" and parse_verdict("no idea") == "TIE"
    rows = [{"id": "d", "condition": "chat", "model": m, "persona_id": 0, "response": r}
            for m, r in (("good", "long helpful answer"), ("bad", "no"))]
    judge = lambda msgs: "A" if "Response A: long" in msgs[0]["content"] else "B"
    df = judge_pairs(rows, {"d": {"history": ["hi"]}}, {0: {"summary": "x"}}, judge)
    assert win_matrix(df).loc["good", "bad"] == 1 and wlt(df).loc["good", "wins"] == 1


def test_end_to_end_offline(tmp_path):
    sel, per, resp = tmp_path / "sel.jsonl", tmp_path / "personas.json", tmp_path / "resp.jsonl"
    select_main(["--input", str(S / "conversations.jsonl"), "--output", str(sel), "--quantile", "0.0"])
    personas_main(["--input", str(sel), "--output", str(per), "--embedder", "hash", "--min-cluster-size", "4",
                   "--umap-dims", "3"])
    meta = json.loads(per.read_text())
    assert meta["personas"], "expected at least one persona cluster"
    gen_main(["--dialogues", str(S / "dialogues.jsonl"), "--personas", str(per), "--models", "echo",
              "--output", str(resp)])
    rows = [json.loads(l) for l in resp.read_text().splitlines()]
    assert len(rows) == 6 * 3 and {r["condition"] for r in rows} == {"chat", "persona", "persona+chat"}
    gen_main(["--dialogues", str(S / "dialogues.jsonl"), "--personas", str(per), "--models", "echo",
              "--output", str(resp)])  # resume: nothing new appended
    assert len(resp.read_text().splitlines()) == 18
    eval_main(["--responses", str(resp), "--out-dir", str(tmp_path / "eval"), "--no-bertscore"])
    assert (tmp_path / "eval" / "gains_and_consistency.csv").exists()
