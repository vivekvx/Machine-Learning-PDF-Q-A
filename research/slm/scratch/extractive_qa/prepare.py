"""Fresh synthetic extractive QA corpus with exact char and token spans."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1"
sys.path.insert(0, str(HERE.parent / "real_text"))
from v3_tokenizer import V3ByteBPE  # noqa: E402

PREFIX = {"service_record": "SR", "deployment_sheet": "DS", "operator_log": "OL",
          "maintenance_bulletin": "MB", "inspection_card": "IC"}
INTRO = {"service_record": "Service record", "deployment_sheet": "Deployment sheet",
         "operator_log": "Operator log", "maintenance_bulletin": "Maintenance bulletin",
         "inspection_card": "Inspection card"}
VALUES = {"method": ["index scan", "hash join", "merge sort", "direct lookup", "range scan", "block copy",
                     "stream merge", "tree walk", "linear probe", "batch insert", "graph search", "cache flush"],
          "owner": [f"team {x}" for x in ("amber", "birch", "cobalt", "delta", "elm", "frost", "gold", "harbor",
                                           "iris", "jade", "kelp", "lime")]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(field, split, family, index, salt):
    number = int(hashlib.sha256(f"{split}:{family}:{index}:{field}:{salt}".encode()).hexdigest()[:8], 16)
    if field in VALUES:
        return VALUES[field][number % len(VALUES[field])]
    return f"{10 + number % 900} units" if field == "quantity" else str(2100 + number % 800)


def entity(split, family, index):
    opaque = hashlib.sha256(f"{split}:{family}:{index}:entity".encode()).hexdigest()[:8].upper()
    return f"{PREFIX[family]}-{opaque}"


def line(family, name, details):
    facts = " ; ".join(f"{field} = [ {answer} ]" for field, answer in details)
    return f"{INTRO[family]} {name}: {facts} ."


def question(name, field, family, index):
    forms = [f"For {name}, what is the {field}?", f"Which {field} is recorded for {name}?"]
    return forms[index % 2]


def input_text(passage, question_text):
    return f"Passage:\n{passage}\nQuestion: {question_text}\n"


def make_example(split, family, index, seed):
    rng = random.Random(f"{seed}:{split}:{family}:{index}")
    fields = ["method", "quantity", "port", "owner"]
    asked = fields[(index // 2) % len(fields)]
    label_rng = random.Random(f"{seed}:label:{split}:{family}:{index // 8}:{asked}")
    answerable = (index % 2) == label_rng.randrange(2)
    name = entity(split, family, index)
    distractor = "DX-" + name
    alternatives = [field for field in fields if field != asked]
    rng.shuffle(alternatives)
    chosen = [asked, alternatives[0]] if answerable else alternatives[:2]
    target_details = [(field, value(field, split, family, index, 0)) for field in chosen]
    distractor_details = [(field, value(field, split, family, index, 1)) for field in (asked, alternatives[1])]
    rng.shuffle(target_details)
    rng.shuffle(distractor_details)
    target_line = line(family, name, target_details)
    distractor_line = line(family, distractor, distractor_details)
    passage = "\n".join((target_line, distractor_line) if rng.randrange(2) else (distractor_line, target_line))
    answer = next((v for field, v in target_details if field == asked), None)
    if (answer is not None) != answerable:
        raise ValueError("Answerability construction mismatch")
    char_start = passage.index(target_line) + target_line.index(f"{asked} = [ {answer} ]") + len(f"{asked} = [ ") if answerable else None
    char_end = char_start + len(answer) if answerable else None
    if answerable and passage[char_start:char_end] != answer:
        raise ValueError("Span text mismatch")
    return {"id": f"{split}-{family}-{index:04d}", "split": split, "family": family,
            "entity": name, "distractor_entity": distractor,
            "fact_id": f"{split}:{family}:{name}:{asked}", "asked_field": asked,
            "passage": passage, "question": question(name, asked, family, index),
            "answerable": answerable, "expected_answer": answer if answerable else "UNANSWERABLE",
            "answer_char_start": char_start, "answer_char_end": char_end,
            "target_line": target_line, "distractor_line": distractor_line,
            "source": "deterministic_original_synthetic_template"}


def tokenized(tok, row):
    text = input_text(row["passage"], row["question"])
    ids = tok.encode(text)
    passage_start = len(tok.encode("Passage:\n"))
    passage_end = len(tok.encode("Passage:\n" + row["passage"]))
    if row["answerable"]:
        offset = len("Passage:\n")
        start = len(tok.encode(text[:offset + row["answer_char_start"]]))
        end_exclusive = len(tok.encode(text[:offset + row["answer_char_end"]]))
        if tok.decode(ids[start:end_exclusive]) != row["expected_answer"]:
            raise ValueError(f"Answer span not aligned to tokenizer: {row['id']}")
        if not (passage_start <= start < end_exclusive <= passage_end):
            raise ValueError("Answer span outside passage")
        end = end_exclusive - 1
    else:
        start = end = None
    return {"ids": ids, "passage_start": passage_start, "passage_end": passage_end,
            "answer_start_token": start, "answer_end_token": end}


def build():
    if (V1 / "manifest.json").exists():
        raise FileExistsError("Extractive v1 manifest exists")
    config = json.loads((V1 / "config.json").read_text())
    splits = {split: [make_example(split, family, i, config["seed"])
                      for family in families for i in range(config["examples_per_family"][split])]
              for split, families in config["families_by_split"].items()}
    all_rows = [r for subset in splits.values() for r in subset]
    if len({r["fact_id"] for r in all_rows}) != len(all_rows) or len({r["entity"] for r in all_rows}) != len(all_rows):
        raise ValueError("Entity or fact leakage")
    for split, subset in splits.items():
        counts = Counter((r["asked_field"], r["answerable"]) for r in subset)
        if len(set(counts.values())) != 1:
            raise ValueError(f"Field/label balance failed: {split}")
    tok = V3ByteBPE.train([input_text(r["passage"], r["question"]) for r in splits["train"]],
                          config["tokenizer_target_size"])
    if tok.vocab_size != config["tokenizer_target_size"]:
        raise ValueError("Tokenizer vocabulary target not reached")
    stats = {}
    for split, subset in splits.items():
        lengths = []
        for row in subset:
            encoded = tokenized(tok, row)
            if tok.decode(encoded["ids"]) != input_text(row["passage"], row["question"]):
                raise ValueError("Tokenizer roundtrip failed")
            if len(encoded["ids"]) > config["training"]["maximum_sequence_length"]:
                raise ValueError(f"Input too long: {row['id']}")
            lengths.append(len(encoded["ids"]))
        stats[split] = {"count": len(subset), "answerable": sum(r["answerable"] for r in subset),
                        "unanswerable": sum(not r["answerable"] for r in subset),
                        "min_tokens": min(lengths), "max_tokens": max(lengths),
                        "mean_tokens": sum(lengths) / len(lengths)}
    for split, subset in splits.items():
        path = V1 / ("test.sealed.jsonl" if split == "test" else f"{split}.jsonl")
        with path.open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    (V1 / "tokenizer.json").write_text(json.dumps({"type": "train_only_byte_bpe", "target_size": config["tokenizer_target_size"],
                                                  "training_ids": [r["id"] for r in splits["train"]],
                                                  "merges": tok.merges}, separators=(",", ":")) + "\n")
    names = ["config.json", "train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json"]
    manifest = {"schema_version": 1, "source": "original deterministic synthetic technical templates; no external content",
                "stats": stats, "split_families": config["families_by_split"],
                "split_ids_sha256": {split: hashlib.sha256("\n".join(r["id"] for r in subset).encode()).hexdigest()
                                     for split, subset in splits.items()},
                "tokenizer_fit_split": "train", "test_sealed_for_selection": True,
                "file_sha256": {name: sha(V1 / name) for name in names}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stats


def verify():
    manifest = json.loads((V1 / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha(V1 / name) != expected:
            raise ValueError(f"Extractive artifact changed: {name}")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.build == args.verify:
        parser.error("Choose exactly one action")
    print(json.dumps(build() if args.build else verify(), indent=2))
