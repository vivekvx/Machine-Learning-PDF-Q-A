"""Deterministic synthetic extractive QA with family- and fact-disjoint splits."""
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = os.environ.get("SLMA_QA_VERSION", "v1")
if VERSION not in ("v1", "v2", "v3"):
    raise ValueError("Unknown synthetic QA corpus version")
V1 = HERE / VERSION
sys.path.insert(0, str(HERE.parent / "real_text"))
from v3_tokenizer import V3ByteBPE, SPECIAL  # noqa: E402

VALUES = {
    "mode": ["read-only", "write-through", "append-only", "strict", "relaxed", "buffered", "direct", "staged"],
    "owner": ["team amber", "team birch", "team cobalt", "team delta", "team elm", "team frost", "team gold", "team harbor"],
}
FAMILY_PREFIX = {"manual": "TM", "deployment": "TD", "operator_note": "TO", "bulletin": "VB", "field_card": "XF"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_value(field, index, split, salt):
    if field in VALUES:
        return VALUES[field][(index * 5 + salt * 3) % len(VALUES[field])]
    base = {"train": 20, "validation": 400, "test": 700}[split]
    number = base + ((index * 37 + salt * 19) % 260)
    return f"{number} seconds" if field == "timeout" else str(2000 + number)


def clause(field, value, family):
    if family == "manual":
        return f"{field}: {value}"
    if family == "deployment":
        return f"{field} {value}"
    if family == "operator_note":
        return f"{field} = {value}"
    if family == "bulletin":
        return f"{field} {value}"
    return f"{field}: {value}"


def sentence(name, details, family):
    facts = " and ".join(clause(field, value, family) for field, value in details)
    if family == "manual":
        return f"The technical manual lists {name} with {facts}."
    if family == "deployment":
        return f"Deployment settings for {name} specify {facts}."
    if family == "operator_note":
        return f"For {name}, the operator recorded {facts}."
    if family == "bulletin":
        return f"A maintenance bulletin describes {name} as having {facts}."
    return f"The inspection card for {name} reports {facts}."


def question(name, field, family, index):
    forms = {
        "manual": [f"What is the {field} for {name}?", f"According to the manual, give {name}'s {field}."],
        "deployment": [f"Which {field} is set for {name}?", f"Give the {field} configured on {name}."],
        "operator_note": [f"What {field} was recorded for {name}?", f"Identify {name}'s {field}."],
        "bulletin": [f"What {field} does the bulletin report for {name}?", f"State the reported {field} of {name}."],
        "field_card": [f"From the card, what is {name}'s {field}?", f"Report the {field} shown for {name}."],
    }
    return forms[family][index % 2]


def make_example(split, family, index, seed):
    rng = random.Random(f"{seed}:{split}:{family}:{index}")
    fields = ["timeout", "port", "mode", "owner"]
    # v1 preserved for audit. v2 removes the v1 parity/field shortcut: every
    # requested field occurs equally often with both answerability labels.
    asked = fields[(index // 2) % len(fields)] if VERSION in ("v2", "v3") else fields[index % len(fields)]
    if VERSION == "v3":
        # Each field pair has one positive and one negative, while the label
        # bit and opaque entity name cannot be read from the example index.
        label_rng = random.Random(f"{seed}:label:{split}:{family}:{index // 8}:{asked}")
        answerable = (index % 2) == label_rng.randrange(2)
        opaque = hashlib.sha256(f"{seed}:entity:{split}:{family}:{index}".encode()).hexdigest()[:8].upper()
        target_name = f"{FAMILY_PREFIX[family]}-{opaque}"
        distractor_name = f"DX-{FAMILY_PREFIX[family]}-{opaque}"
    else:
        answerable = index % 2 == 0
        target_name = f"{FAMILY_PREFIX[family]}-{index:04d}"
        distractor_name = f"DX-{FAMILY_PREFIX[family]}-{index:04d}"
    other = fields[(fields.index(asked) + 1 + (index // 4) % 3) % len(fields)]
    target_fields = [asked, other] if answerable else [other]
    target_details = [(field, field_value(field, index, split, fields.index(field))) for field in target_fields]
    distractor_fields = [asked, fields[(fields.index(asked) + 2) % len(fields)]]
    distractor_details = [(field, field_value(field, index + 17, split, fields.index(field) + 4)) for field in distractor_fields]
    if rng.randrange(2):
        target_details.reverse()
    if rng.randrange(2):
        distractor_details.reverse()
    target_sentence = sentence(target_name, target_details, family)
    distractor_sentence = sentence(distractor_name, distractor_details, family)
    passage = " ".join((target_sentence, distractor_sentence) if rng.randrange(2) else (distractor_sentence, target_sentence))
    expected = next((value for field, value in target_details if field == asked), "UNANSWERABLE")
    if answerable != (expected != "UNANSWERABLE"):
        raise ValueError("Answerability construction failed")
    if answerable and expected not in target_sentence:
        raise ValueError("Correct answer is absent from target evidence")
    if not answerable and asked in [field for field, _ in target_details]:
        raise ValueError("Unanswerable target contains asked field")
    return {"id": f"{split}-{family}-{index:04d}", "split": split, "family": family,
            "fact_id": f"{split}:{target_name}:{asked}", "target_entity": target_name,
            "distractor_entity": distractor_name, "asked_field": asked,
            "passage": passage, "question": question(target_name, asked, family, index),
            "expected_answer": expected, "answerable": answerable,
            "evidence_sentence": target_sentence if answerable else None,
            "target_sentence": target_sentence, "distractor_sentence": distractor_sentence,
            "source": "deterministic_synthetic_template_v1"}


def prompt(passage, question):
    return f"Passage: {passage}\nQuestion: {question}\nAnswer: "


def answer_sequence(tokenizer, row):
    prefix = tokenizer.encode(prompt(row["passage"], row["question"]))
    answer = tokenizer.encode(row["expected_answer"])
    full = prefix + answer + [SPECIAL["eos"]]
    # Loss at next-token positions for the answer and its EOS only.
    mask = [0] * (len(prefix) - 1) + [1] * (len(answer) + 1)
    if len(mask) != len(full) - 1 or sum(mask) != len(answer) + 1:
        raise ValueError("Answer-only loss mask is inconsistent")
    return full, mask, len(prefix), len(answer)


def build():
    if (V1 / "manifest.json").exists():
        raise FileExistsError("QA v1 artifacts already exist")
    config = json.loads((V1 / "config.json").read_text())
    rows = {}
    for split, families in config["families_by_split"].items():
        rows[split] = [make_example(split, family, index, config["seed"])
                       for family in families for index in range(config["examples_per_family"][split])]
    if any({r["family"] for r in rows[a]} & {r["family"] for r in rows[b]}
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Template family leakage")
    if len({r["fact_id"] for subset in rows.values() for r in subset}) != sum(map(len, rows.values())):
        raise ValueError("Generated fact leakage")
    if VERSION in ("v2", "v3"):
        for split, subset in rows.items():
            balance = Counter((r["asked_field"], r["answerable"]) for r in subset)
            if len(set(balance.values())) != 1:
                raise ValueError(f"Field/answerability imbalance in {split}")
    tokenizer = V3ByteBPE.train([prompt(r["passage"], r["question"]) + r["expected_answer"] for r in rows["train"]],
                                config["tokenizer_target_size"])
    if tokenizer.vocab_size != 1024:
        raise ValueError("Vocabulary target not reached")
    lengths = {}
    for split, subset in rows.items():
        seq_lengths = []
        for row in subset:
            full, mask, prefix_len, answer_len = answer_sequence(tokenizer, row)
            if tokenizer.decode(full[:-1]) != prompt(row["passage"], row["question"]) + row["expected_answer"]:
                raise ValueError("Tokenizer roundtrip failed")
            if len(full) > config["training"]["maximum_sequence_length"]:
                raise ValueError(f"QA sequence too long in {split}")
            seq_lengths.append(len(full))
        lengths[split] = {"count": len(subset), "min_sequence_ids": min(seq_lengths),
                          "max_sequence_ids": max(seq_lengths), "mean_sequence_ids": sum(seq_lengths) / len(seq_lengths),
                          "answerable": sum(r["answerable"] for r in subset),
                          "unanswerable": sum(not r["answerable"] for r in subset)}
    for split, subset in rows.items():
        name = "test.sealed.jsonl" if split == "test" else f"{split}.jsonl"
        with (V1 / name).open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    tokdata = {"schema_version": 1, "type": "byte_bpe_from_synthetic_qa_train_only",
               "target_size": config["tokenizer_target_size"], "vocab_size": tokenizer.vocab_size,
               "training_ids": [r["id"] for r in rows["train"]], "special_ids": SPECIAL, "merges": tokenizer.merges}
    (V1 / "tokenizer.json").write_text(json.dumps(tokdata, separators=(",", ":")) + "\n")
    names = ("config.json", "train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json")
    manifest = {"schema_version": 1, "source": "deterministic generated technical templates; no external text",
                "split_families": config["families_by_split"],
                "split_ids": {split: [r["id"] for r in subset] for split, subset in rows.items()},
                "split_fact_id_sha256": {split: hashlib.sha256("\n".join(r["fact_id"] for r in subset).encode()).hexdigest()
                                         for split, subset in rows.items()},
                "stats": lengths, "test_sealed_for_model_selection": True,
                "tokenizer_fit_split": "train", "file_sha256": {name: digest(V1 / name) for name in names}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"stats": lengths, "vocab_size": tokenizer.vocab_size}


def verify():
    manifest = json.loads((V1 / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if digest(V1 / name) != expected:
            raise ValueError(f"QA corpus artifact hash mismatch: {name}")
    return {"status": "verified", "stats": manifest["stats"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.build == args.verify:
        parser.error("Choose --build or --verify")
    print(json.dumps(build() if args.build else verify(), indent=2))
