"""Create a separate FAISS-retrieved validation condition from the same questions.

This script selects passages from documents using question text only. It does not
read gold evidence or reference answers until after retrieval has selected IDs.
"""
import json
from pathlib import Path

from build_eval import ROOT, OUT, rows, MAX_CHARS_PER_PASSAGE
from curate_dataset import write_jsonl

RETRIEVAL_METHOD = "per-document word-and-bigram TF-IDF; FAISS IndexFlatIP"


def build():
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    gold_rows = list(rows(ROOT / "validation_gold_eval.jsonl"))
    docs = {d["document_id"]: d for d in rows(OUT / "validation_documents.jsonl")}
    output = []
    for row in gold_rows:
        doc = docs[row["document_id"]]
        paragraphs = [(s["section_index"], pi, text)
                      for s in doc["sections"] for pi, text in enumerate(s["paragraphs"])
                      if text and text.strip()]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8192, norm="l2")
        vectors = vectorizer.fit_transform([p[2] for p in paragraphs]).toarray().astype("float32")
        query = vectorizer.transform([row["question"]]).toarray().astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, indices = index.search(query, min(3, len(paragraphs)))
        selected = []
        for score, i in zip(scores[0], indices[0]):
            section, pi, text = paragraphs[int(i)]
            selected.append({"evidence_id": f"retrieved:{section}:{pi}",
                             "section_index": section, "paragraph_index": pi,
                             "text": text[:MAX_CHARS_PER_PASSAGE], "truncated": len(text) > MAX_CHARS_PER_PASSAGE,
                             "retrieval_score": float(score), "source_type": "faiss_retrieved_paragraph"})
        output.append({**row, "condition": "faiss_retrieved_evidence",
                       "selected_passages": selected,
                       "retrieval_method": RETRIEVAL_METHOD})
    write_jsonl(ROOT / "validation_faiss_eval.jsonl", output)
    return output


if __name__ == "__main__":
    print(f"Wrote {len(build())} FAISS validation inputs")
