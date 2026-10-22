"""Step 2: cluster utterances into emergent personas (Sentence-BERT -> UMAP -> HDBSCAN) and
summarise each cluster into a short persona profile.

Each persona keeps: top keywords (class-based TF-IDF), representative utterances (closest to
the cluster centroid), simple style cues, its centroid embedding, and a natural-language
summary. With ``--summarizer`` an LLM writes the summary; otherwise a template is used.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from .embed import DEFAULT_MODEL, Embedder
from .select import read_jsonl


def reduce(emb: np.ndarray, n_components: int = 5, n_neighbors: int = 15, seed: int = 42) -> np.ndarray:
    """UMAP as in the paper; falls back to PCA if ``umap-learn`` is not installed."""
    n_components = min(n_components, emb.shape[1], max(2, emb.shape[0] - 2))
    try:
        import umap
        return umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, len(emb) - 1),
                         metric="cosine", random_state=seed).fit_transform(emb)
    except ImportError:
        warnings.warn("umap-learn not installed; using PCA for dimensionality reduction")
        return PCA(n_components=n_components, random_state=seed).fit_transform(emb)


def cluster(reduced: np.ndarray, min_cluster_size: int = 15, min_samples: Optional[int] = None) -> np.ndarray:
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(reduced)


def keywords_per_cluster(texts: List[str], labels: np.ndarray, top_k: int = 8) -> Dict[int, List[str]]:
    """Class-based TF-IDF: words frequent in one cluster but rare across clusters."""
    ids = sorted(set(labels) - {-1})
    if not ids:
        return {}
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in ids]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tf = cv.fit_transform(docs).toarray().astype(float)
    tf = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    idf = np.log(1 + tf.shape[0] / np.maximum((tf > 0).sum(0), 1))
    vocab = np.array(cv.get_feature_names_out())
    return {c: vocab[np.argsort(-(tf[i] * idf))[:top_k]].tolist() for i, c in enumerate(ids)}


def style_cues(texts: List[str]) -> Dict[str, float]:
    n = max(len(texts), 1)
    return {"question_rate": round(sum("?" in t for t in texts) / n, 3),
            "exclamation_rate": round(sum("!" in t for t in texts) / n, 3),
            "politeness_rate": round(sum(any(w in t.lower() for w in ("please", "thank", "could you")) for t in texts) / n, 3),
            "avg_words": round(float(np.mean([len(t.split()) for t in texts])), 1) if texts else 0.0}


def template_summary(kw: List[str], cues: Dict[str, float]) -> str:
    tone = []
    if cues["question_rate"] > 0.5:
        tone.append("inquisitive")
    if cues["politeness_rate"] > 0.3:
        tone.append("polite")
    if cues["exclamation_rate"] > 0.3:
        tone.append("enthusiastic")
    length = "detailed, long" if cues["avg_words"] > 40 else "short, direct"
    return (f"A {', '.join(tone) or 'neutral'} user who writes {length} messages, mostly about "
            f"{', '.join(kw[:5])}.")


def llm_summary(generate, kw: List[str], examples: List[str]) -> str:
    prompt = ("Below are messages written by one group of chatbot users.\n"
              f"Frequent topics: {', '.join(kw)}\nExample messages:\n- " + "\n- ".join(examples) +
              "\n\nIn 2-3 sentences, describe this group as a single user persona: their interests, "
              "the kind of questions they ask, and their tone. Reply with the description only.")
    return generate([{"role": "user", "content": prompt}]).strip()


def build_personas(texts: List[str], emb: np.ndarray, labels: np.ndarray, n_examples: int = 5,
                   generate=None) -> List[Dict]:
    kws = keywords_per_cluster(texts, labels)
    personas = []
    for c in sorted(kws):
        idx = np.where(labels == c)[0]
        centroid = emb[idx].mean(0)
        order = idx[np.argsort(-(emb[idx] @ centroid))]
        examples = [texts[i] for i in order[:n_examples]]
        cues = style_cues([texts[i] for i in idx])
        summary = llm_summary(generate, kws[c], examples) if generate else template_summary(kws[c], cues)
        personas.append({"id": int(c), "size": int(len(idx)), "keywords": kws[c], "style": cues,
                         "examples": examples, "summary": summary,
                         "centroid": [round(float(x), 6) for x in centroid]})
    return personas


def main(argv=None):
    from .llm import make_client

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL from userprint.select")
    ap.add_argument("--output", required=True, help="personas JSON")
    ap.add_argument("--embedder", default=DEFAULT_MODEL, help="Sentence-BERT model, or 'hash' for a quick offline run")
    ap.add_argument("--umap-dims", type=int, default=5)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--summarizer", default=None, help="LLM for persona summaries, e.g. ollama:gemma3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    texts = [r["text"] for r in read_jsonl(args.input)]
    emb = Embedder(args.embedder).encode(texts)
    labels = cluster(reduce(emb, args.umap_dims, seed=args.seed), args.min_cluster_size)
    gen = make_client(args.summarizer) if args.summarizer else None
    personas = build_personas(texts, emb, labels, generate=gen)
    out = {"embedder": args.embedder, "n_utterances": len(texts),
           "noise": int((labels == -1).sum()), "personas": personas}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(personas)} personas from {len(texts):,} utterances ({out['noise']:,} unclustered)")
    for p in personas:
        print(f"  #{p['id']:<3} n={p['size']:<5} {', '.join(p['keywords'][:5])}")


if __name__ == "__main__":
    main()
