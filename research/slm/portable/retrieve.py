"""Build distinct TF-IDF+FAISS or MiniLM+FAISS validation conditions, offline."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data/v1"
TOKENIZER = ROOT / ".cache/qwen3-0.6b-4bit"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def paragraphs(doc):
    return [(s["section_index"], i, text) for s in doc["sections"]
            for i, text in enumerate(s["paragraphs"]) if text and text.strip()]


def tfidf_rank(question, items, top_k):
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8192, norm="l2")
    vectors = vectorizer.fit_transform([p[2] for p in items]).toarray().astype("float32")
    query = vectorizer.transform([question]).toarray().astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, indices = index.search(query, min(top_k, len(items)))
    return [(int(i), float(score)) for i, score in zip(indices[0], scores[0])]


def minilm_model():
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("MiniLM condition requires a working PyTorch/sentence-transformers environment") from exc

    try:
        path = snapshot_download(MINILM_ID, revision=MINILM_REV,
                                 local_files_only=True)
    except Exception as exc:
        raise RuntimeError("Pinned MiniLM snapshot is not fully cached; no automatic download is allowed") from exc
    return SentenceTransformer(path)


def minilm_rank(question, items, top_k, model):
    import faiss
    import numpy as np

    vectors = model.encode([p[2] for p in items], batch_size=8,
                           normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    query = model.encode([question], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    scores, indices = index.search(np.ascontiguousarray(query), min(top_k, len(items)))
    return [(int(i), float(score)) for i, score in zip(indices[0], scores[0])]


def prompt_tokens(row, passages, tokenizer, experiment):
    body = "\n".join(f"Passage [{p['passage_id']}]: {p['text']}" for p in passages)
    if not body:
        body = "No passage fit the configured context."
    messages = [{"role": "system", "content": experiment["system_instruction"]},
                {"role": "user", "content":
                 f"Document: {row['document_id']} — {row['title']}\n"
                 f"{body}\n\nQuestion: {row['question']}"}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             enable_thinking=False)
    ids = rendered["input_ids"] if isinstance(rendered, dict) else rendered
    return messages, len(ids)


def pack(row, items, ranking, tokenizer, experiment):
    selected, audit = [], []
    for index, score in ranking:
        section, paragraph, text = items[index]
        passage_id = f"section:{section}:paragraph:{paragraph}"
        passage = {"passage_id": passage_id, "section_index": section,
                   "paragraph_index": paragraph, "text": text,
                   "sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "source_type": "retrieved_document_paragraph", "retrieval_score": score}
        _, length = prompt_tokens(row, selected + [passage], tokenizer, experiment)
        included = length + experiment["maximum_generation_tokens"] <= experiment["maximum_sequence_tokens"]
        audit.append({"passage_id": passage_id, "score": score, "included": included,
                      "reason": "fits" if included else "exceeds_token_budget_without_truncation",
                      "source_sha256": passage["sha256"]})
        if included:
            selected.append(passage)
    messages, length = prompt_tokens(row, selected, tokenizer, experiment)
    if length + experiment["maximum_generation_tokens"] > experiment["maximum_sequence_tokens"]:
        raise RuntimeError(f"Even empty retrieval context exceeds budget: {row['question_id']}")
    gold = row["selected_passages"][0]
    gold_id = gold["passage_id"] if row["answerability"] == "answerable" else None
    return {**row, "messages": messages + [{"role": "assistant", "content": row["target_answer"]}],
            "selected_passages": selected, "retrieval_candidates": audit,
            "retrieval_failure": "none_fit" if not selected else None,
            "gold_passage_retrieved": gold_id in {p["passage_id"] for p in selected} if gold_id else None,
            "prompt_tokens": length, "full_tokens": None,
            "completion_tokens": None}


def build(condition, extra_site_packages=None):
    if condition not in ("tfidf_faiss", "minilm_faiss"):
        raise ValueError("Unknown retrieval condition")
    if extra_site_packages:
        extra = Path(extra_site_packages).resolve()
        if not extra.is_dir():
            raise FileNotFoundError(extra)
        sys.path.append(str(extra))
    config = json.loads((HERE / "v1/retrieval.json").read_text())
    experiment = json.loads((HERE / "v1/experiment.json").read_text())
    source = list(rows(DATA / "validation_gold.jsonl"))
    docs = {d["document_id"]: d for d in rows(ROOT / "curated/validation_documents.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    model = minilm_model() if condition == "minilm_faiss" else None
    output = []
    for row in source:
        doc = docs[row["document_id"]]
        items = paragraphs(doc)
        top_k = config["conditions"][condition]["top_k"]
        ranking = (tfidf_rank(row["question"], items, top_k) if model is None else
                   minilm_rank(row["question"], items, top_k, model))
        result = pack({**row, "title": doc["title"]}, items, ranking, tokenizer, experiment)
        result["condition"] = condition
        output.append(result)
    path = DATA / f"validation_{condition}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {"condition": condition, "source_sha256": sha(DATA / "validation_gold.jsonl"),
            "retrieval_config_sha256": sha(HERE / "v1/retrieval.json"),
            "tokenizer_config_sha256": sha(TOKENIZER / "tokenizer_config.json"),
            "output_sha256": sha(path), "rows": len(output),
            "rows_without_fitting_passage": sum(not x["selected_passages"] for x in output),
            "answerable_gold_recall_count": sum(x["gold_passage_retrieved"] is True for x in output),
            "candidate_passages_skipped_for_budget": sum(not p["included"] for x in output for p in x["retrieval_candidates"]),
            "model_revision": MINILM_REV if condition == "minilm_faiss" else None,
            "software_versions": {name: importlib.metadata.version(name) for name in
                                  (("faiss-cpu", "scikit-learn", "sentence-transformers")
                                   if condition == "minilm_faiss" else ("faiss-cpu", "scikit-learn"))},
            "no_test_answers_read": True}
    (DATA / f"validation_{condition}.manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("tfidf_faiss", "minilm_faiss"), required=True)
    parser.add_argument("--extra-site-packages", type=Path,
                        help="Optional existing environment with FAISS/sklearn dependencies")
    args = parser.parse_args()
    try:
        result = build(args.condition, args.extra_site_packages)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2))
