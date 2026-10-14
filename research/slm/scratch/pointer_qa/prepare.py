"""Original answerable-only technical facts with document-disjoint splits."""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1"
sys.path.insert(0, str(HERE.parent))
from real_text.v3_tokenizer import V3ByteBPE  # noqa: E402
from working_qa.prepare import COLORS, DEVICES, enumerate_candidates  # noqa: E402

FIELDS = ("method", "metric", "capacity", "owner")
SYNONYMS = {"method": ("method", "procedure", "technique"),
            "metric": ("metric", "performance measure", "benchmark"),
            "capacity": ("capacity", "storage size", "limit"),
            "owner": ("owner", "assigned team", "responsible group")}
METHODS = ("index scan", "hash join", "merge sort", "direct lookup", "range scan",
           "block copy", "stream merge", "tree walk", "linear probe", "batch insert")
FAMILIES = {"manual": "Manual note", "deployment": "Deployment record",
            "logbook": "Logbook entry", "specification": "Specification sheet",
            "bulletin": "Service bulletin", "inspection": "Inspection record"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pools(seed):
    names = [f"{color} {device}" for color in COLORS for device in DEVICES]
    random.Random(seed).shuffle(names)
    result = {"train": names[:60], "validation": names[60:78], "test": names[78:]}
    train_words = set(" ".join(result["train"]).split())
    if any(not set(name.split()) <= train_words for name in result["validation"] + result["test"]):
        raise ValueError("Held-out entity component absent from train vocabulary")
    return result


def value(field, rng, used):
    while True:
        if field == "method":
            item = rng.choice(METHODS)
        elif field == "metric":
            item = f"{rng.randrange(15, 400)} ms"
        elif field == "capacity":
            item = f"{rng.randrange(16, 900)} units"
        else:
            item = f"team {rng.choice(COLORS)}"
        if item not in used:
            used.add(item)
            return item


def sentence(family, entity, field, item, style):
    tag = FAMILIES[family]
    return (
        f"For {entity}, the {field} is [ {item} ].",
        f"{tag} records {entity}'s {field} as [ {item} ].",
        f"{entity} uses [ {item} ] for the {field}, according to the {tag.lower()}.",
        f"The {field} assigned to {entity} is [ {item} ] in this {tag.lower()}."
    )[style]


def document(split, family, index, names, seed):
    rng = random.Random(f"{seed}:{split}:{family}:{index}")
    # Distinct full names but shared color/device components across splits.
    entities = rng.sample(names, 2)
    fields = rng.sample(FIELDS, 2)
    used_values = set()
    facts = [(entity, field, value(field, rng, used_values))
             for entity in entities for field in fields]
    rng.shuffle(facts)
    forms = [rng.randrange(3) if split == "train" else rng.randrange(1, 4) for _ in facts]
    sentences = [sentence(family, entity, field, item, form)
                 for (entity, field, item), form in zip(facts, forms)]
    passage = " ".join(sentences)
    candidates = []
    cursor = 0
    for (entity, field, item), text, form in zip(facts, sentences, forms):
        offset = passage.find(text, cursor)
        if offset < 0:
            raise ValueError("Sentence offset failed")
        match = re.search(r"\[ ([^\[\]\n]+?) \]", text)
        if match is None or match.group(1) != item:
            raise ValueError("Answer span missing")
        candidates.append({"entity": entity, "field": field, "value": item,
                           "start": offset + match.start(1), "end": offset + match.end(1),
                           "sentence_start": offset, "sentence_end": offset + len(text),
                           "sentence_style": form, "candidate_index": len(candidates)})
        cursor = offset + len(text)
    parsed = enumerate_candidates(passage)
    if [(x["start"], x["end"]) for x in parsed] != [(x["start"], x["end"]) for x in candidates]:
        raise ValueError("Candidate parser and source annotations disagree")
    doc_id = f"{split}-{family}-doc-{index:04d}"
    result = []
    for fact_index, candidate in enumerate(candidates):
        synonym_index = rng.randrange(3)
        synonym = SYNONYMS[candidate["field"]][synonym_index]
        forms = (f"What is the {synonym} for {candidate['entity']}?",
                 f"Which {synonym} is recorded for {candidate['entity']}?",
                 f"For {candidate['entity']}, state the {synonym}.")
        question = forms[rng.randrange(3)]
        result.append({"id": f"{doc_id}-q{fact_index}", "document_id": doc_id,
                       "split": split, "family": family, "source": "original_synthetic_pointer_qa_v1",
                       "passage": passage, "question": question,
                       "target_entity": candidate["entity"], "asked_field": candidate["field"],
                       "question_paraphrase": synonym_index > 0,
                       "answerable": True, "expected_answer": candidate["value"],
                       "gold_candidate_index": fact_index,
                       "answer_start": candidate["start"], "answer_end": candidate["end"],
                       "facts": candidates})
    return result


def build():
    if (V1 / "manifest.json").exists():
        raise FileExistsError("Pointer v1 dataset already frozen")
    c = json.loads((V1 / "config.json").read_text())
    names = pools(c["seed"])
    splits = {split: [row for family in families
                      for index in range(c["documents_per_family"][split])
                      for row in document(split, family, index, names[split], c["seed"])]
              for split, families in c["families_by_split"].items()}
    train, validation = splits["train"], splits["validation"]
    if len({row["document_id"] for row in train}) != len(train) // 4:
        raise ValueError("Document count mismatch")
    all_rows = [row for subset in splits.values() for row in subset]
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Question ID duplicate")
    if len({(row["passage"], row["question"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Exact QA duplicate")
    tok = V3ByteBPE.train([f"Question: {row['question']}\nPassage: {row['passage']}"
                           for row in train], c["tokenizer_target_size"])
    if tok.vocab_size != c["tokenizer_target_size"]:
        raise ValueError("Tokenizer target size not reached")
    stats = {}
    for split, subset in splits.items():
        lengths = []
        for row in subset:
            text = f"Question: {row['question']}\nPassage: {row['passage']}"
            ids = tok.encode(text)
            if tok.decode(ids) != text:
                raise ValueError("Tokenizer roundtrip failed")
            lengths.append(len(ids))
            for item in row["facts"]:
                if row["passage"][item["start"]:item["end"]] != item["value"]:
                    raise ValueError("Invalid answer character offsets")
            if row["passage"][row["answer_start"]:row["answer_end"]] != row["expected_answer"]:
                raise ValueError("Gold span misaligned")
        stats[split] = {"documents": len({row["document_id"] for row in subset}),
                        "questions": len(subset), "answerable": len(subset),
                        "min_tokens": min(lengths), "mean_tokens": sum(lengths) / len(lengths),
                        "max_tokens": max(lengths),
                        "answer_position_counts": dict(Counter(row["gold_candidate_index"] for row in subset)),
                        "paraphrase_questions": sum(row["question_paraphrase"] for row in subset)}
    for split, subset in splits.items():
        destination = V1 / ("test.sealed.jsonl" if split == "test" else f"{split}.jsonl")
        with destination.open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    (V1 / "tokenizer.json").write_text(json.dumps({"schema_version": 1,
                                                   "type": "train_only_byte_bpe",
                                                   "target_size": tok.target_size,
                                                   "training_document_ids": sorted({r["document_id"] for r in train}),
                                                   "merges": tok.merges}, separators=(",", ":")) + "\n")
    files = ("config.json", "train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json")
    manifest = {"schema_version": 1, "rights": "original generated synthetic text; no third-party source",
                "stats": stats, "families_by_split": c["families_by_split"],
                "entity_pool_sha256": {key: hashlib.sha256("\n".join(value).encode()).hexdigest()
                                       for key, value in names.items()},
                "tokenizer_fit_split": "train", "test_sealed": True,
                "files_sha256": {name: sha(V1 / name) for name in files}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stats


def verify():
    manifest = json.loads((V1 / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha(V1 / name) != expected:
            raise ValueError(f"Frozen dataset artifact changed: {name}")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build() if args.build else verify(), indent=2))
