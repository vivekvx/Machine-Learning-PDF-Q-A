"""Versioned original synthetic evidence QA with candidate spans and NULL labels."""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1"
sys.path.insert(0, str(HERE.parent / "real_text"))
from v3_tokenizer import V3ByteBPE  # noqa: E402

COLORS = ("amber", "birch", "cobalt", "delta", "elm", "frost", "gold", "harbor", "iris", "jade", "kelp", "lime")
DEVICES = ("relay", "gateway", "node", "buffer", "router", "sensor", "module", "engine")
FIELDS = ("method", "quantity", "port", "owner")
SYNONYMS = {"method": ("method", "procedure", "technique"),
            "quantity": ("quantity", "amount", "capacity"),
            "port": ("port", "network port", "endpoint"),
            "owner": ("owner", "responsible team", "assigned team")}
METHODS = ("index scan", "hash join", "merge sort", "direct lookup", "range scan", "block copy",
           "stream merge", "tree walk", "linear probe", "batch insert", "graph search", "cache flush")
OWNERS = tuple(f"team {color}" for color in COLORS)
FAMILY_OPENERS = {"manual": "Manual entry", "deployment": "Deployment note", "logbook": "Logbook entry",
                  "specification": "Specification note", "bulletin": "Service bulletin", "inspection": "Inspection record"}
CANDIDATE = re.compile(r"\[ ([^\[\]\n]+?) \]")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entity_pools(seed):
    names = [f"{color} {device}" for color in COLORS for device in DEVICES]
    random.Random(seed).shuffle(names)
    pools = {"train": names[:60], "validation": names[60:78], "test": names[78:96]}
    train_words = set(" ".join(pools["train"]).split())
    if any(not set(name.split()) <= train_words for name in pools["validation"] + pools["test"]):
        raise ValueError("Held-out entity component absent from training")
    return pools


def fact_value(field, rng):
    if field == "method":
        return rng.choice(METHODS)
    if field == "owner":
        return rng.choice(OWNERS)
    if field == "quantity":
        return f"{rng.randrange(12, 320)} units"
    return str(rng.randrange(2100, 2900))


def fact_sentence(family, entity, field, value, style):
    opener = FAMILY_OPENERS[family]
    forms = (f"{opener} for {entity}: {field} is [ {value} ].",
             f"{opener} says {entity} has {field} [ {value} ].",
             f"For {entity}, the {opener.lower()} lists {field} as [ {value} ].")
    return forms[style % len(forms)]


def question(entity, field, tier, variant):
    synonym = SYNONYMS[field][variant % (3 if tier == "paraphrase_conflict" else 2)]
    forms = (f"What is the {synonym} for {entity}?", f"Which {synonym} is recorded for {entity}?",
             f"State {entity}'s {synonym}.")
    return forms[(variant // 3) % 3]


def make_example(split, family, index, seed, names):
    # Four tiers cycle independently of the balanced label bit. Each adjacent
    # pair within one tier/field has one positive and one negative.
    tier_index = index % 4
    tier = ("single_fact", "numeric", "multi_entity", "paraphrase_conflict")[tier_index]
    ordinal = index // 4
    field = FIELDS[(ordinal // 2) % 4]
    label_bit = random.Random(f"{seed}:label:{split}:{family}:{tier}:{ordinal // 2}").randrange(2)
    answerable = ordinal % 2 == label_bit
    rng = random.Random(f"{seed}:row:{split}:{family}:{index}")
    target = names[(ordinal * 7 + tier_index * 3) % len(names)]
    distractor = names[(ordinal * 11 + tier_index * 5 + 1) % len(names)]
    if distractor == target:
        distractor = names[(names.index(target) + 1) % len(names)]
    other_fields = [x for x in FIELDS if x != field]
    rng.shuffle(other_fields)
    if tier == "numeric":
        field = ("quantity", "port")[(ordinal // 2) % 2]
        other_fields = [x for x in FIELDS if x != field]
        rng.shuffle(other_fields)
    target_facts = [(field, fact_value(field, rng))] if answerable else [(other_fields[0], fact_value(other_fields[0], rng))]
    distractor_facts = []
    if tier != "single_fact":
        distractor_value = fact_value(field, rng)
        if answerable and distractor_value == target_facts[0][1]:
            if field == "method":
                distractor_value = METHODS[(METHODS.index(distractor_value) + 1) % len(METHODS)]
            elif field == "owner":
                distractor_value = OWNERS[(OWNERS.index(distractor_value) + 1) % len(OWNERS)]
            elif field == "quantity":
                distractor_value = f"{int(distractor_value.split()[0]) + 1} units"
            else:
                distractor_value = str(int(distractor_value) + 1)
        distractor_facts.append((field, distractor_value))
    if tier in ("multi_entity", "paraphrase_conflict"):
        target_facts.append((other_fields[1], fact_value(other_fields[1], rng)))
        distractor_facts.append((other_fields[2], fact_value(other_fields[2], rng)))
    facts = [(target, f, v) for f, v in target_facts] + [(distractor, f, v) for f, v in distractor_facts]
    rng.shuffle(facts)
    styles = [rng.randrange(3) for _ in facts]
    sentences = [fact_sentence(family, entity, f, v, style) for (entity, f, v), style in zip(facts, styles)]
    passage = " ".join(sentences)
    candidates = []
    cursor = 0
    for (entity, fact_field, value), sentence in zip(facts, sentences):
        at = passage.find(sentence, cursor)
        if at < 0:
            raise ValueError("Sentence provenance lost")
        match = CANDIDATE.search(sentence)
        if not match or match.group(1) != value:
            raise ValueError("Candidate annotation mismatch")
        candidates.append({"entity": entity, "field": fact_field, "value": value,
                           "start": at + match.start(1), "end": at + match.end(1),
                           "sentence_start": at, "sentence_end": at + len(sentence),
                           "sentence_id": len(candidates)})
        cursor = at + len(sentence)
    correct = [i for i, item in enumerate(candidates) if item["entity"] == target and item["field"] == field]
    if len(correct) != int(answerable):
        raise ValueError("Gold evidence ambiguity")
    answer = candidates[correct[0]]["value"] if answerable else "UNANSWERABLE"
    if answerable and passage[candidates[correct[0]]["start"]:candidates[correct[0]]["end"]] != answer:
        raise ValueError("Gold answer not in passage")
    variant = rng.randrange(9)
    if tier == "paraphrase_conflict":
        variant = 1 + 3 * rng.randrange(3)
    return {"id": f"{split}-{family}-{index:04d}", "split": split, "family": family,
            "tier": tier, "fact_id": f"{split}:{family}:{index}:{target}:{field}",
            "target_entity": target, "asked_field": field, "passage": passage,
            "question": question(target, field, tier, variant), "answerable": answerable,
            "expected_answer": answer, "gold_candidate_index": correct[0] if correct else None,
            "candidates": candidates, "source": "original_deterministic_synthetic_v1"}


def input_text(passage, question_text):
    return f"Question: {question_text}\nPassage: {passage}"


def enumerate_candidates(passage):
    """Find possible value spans only; never select an answer or use a question."""
    found = []
    for sentence in re.finditer(r"[^.]+\.", passage):
        raw = sentence.group()
        left = sentence.start() + len(raw) - len(raw.lstrip())
        clean = passage[left:sentence.end()]
        for match in CANDIDATE.finditer(clean):
            found.append({"value": match.group(1), "start": left + match.start(1),
                          "end": left + match.end(1), "sentence_start": left,
                          "sentence_end": sentence.end(), "sentence_id": len(found)})
    return found


def tokenize(tok, row):
    text = input_text(row["passage"], row["question"])
    ids = tok.encode(text)
    question_end = len(tok.encode(f"Question: {row['question']}\n"))
    passage_offset = len(input_text("", row["question"]))
    mapped = []
    for item in row["candidates"]:
        start = len(tok.encode(text[:passage_offset + item["start"]]))
        end = len(tok.encode(text[:passage_offset + item["end"]]))
        sent_start = len(tok.encode(text[:passage_offset + item["sentence_start"]]))
        sent_end = len(tok.encode(text[:passage_offset + item["sentence_end"]]))
        if tok.decode(ids[start:end]) != item["value"]:
            raise ValueError(f"Candidate token alignment failed: {row['id']}")
        if not (0 <= sent_start <= start < end <= sent_end <= len(ids)):
            raise ValueError("Invalid candidate/sentence positions")
        mapped.append({"start": start, "end": end, "sentence_start": sent_start, "sentence_end": sent_end})
    if tok.decode(ids) != text:
        raise ValueError("Tokenizer roundtrip failed")
    return {"ids": ids, "question_end": question_end, "candidates": mapped}


def build():
    if (V1 / "manifest.json").exists():
        raise FileExistsError("Working QA v1 already exists")
    config = json.loads((V1 / "config.json").read_text())
    names = entity_pools(config["seed"])
    splits = {split: [make_example(split, family, i, config["seed"], names[split])
                      for family in families for i in range(config["examples_per_family"][split])]
              for split, families in config["families_by_split"].items()}
    all_rows = [row for subset in splits.values() for row in subset]
    if len({row["fact_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Fact ID leakage")
    if len({(row["passage"], row["question"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Exact passage/question duplicate")
    for split, subset in splits.items():
        balance = Counter((row["tier"], row["answerable"]) for row in subset)
        if any(abs(balance[tier, True] - balance[tier, False]) > 1 for tier in config["tiers"]):
            raise ValueError(f"Tier label imbalance in {split}")
    tok = V3ByteBPE.train([input_text(row["passage"], row["question"]) for row in splits["train"]],
                          config["tokenizer_target_size"])
    if tok.vocab_size != config["tokenizer_target_size"]:
        raise ValueError("Tokenizer target not reached")
    stats = {}
    for split, subset in splits.items():
        lengths, candidates = [], []
        for row in subset:
            enumerated = enumerate_candidates(row["passage"])
            if [(c["start"], c["end"], c["sentence_start"], c["sentence_end"]) for c in enumerated] != [
                    (c["start"], c["end"], c["sentence_start"], c["sentence_end"]) for c in row["candidates"]]:
                raise ValueError(f"Inference candidate parser differs from annotation: {row['id']}")
            encoded = tokenize(tok, row)
            if len(encoded["ids"]) > config["training"]["maximum_sequence_length"]:
                raise ValueError(f"Sequence over cap: {row['id']}")
            lengths.append(len(encoded["ids"]))
            candidates.append(len(encoded["candidates"]))
        stats[split] = {"examples": len(subset), "answerable": sum(row["answerable"] for row in subset),
                        "unanswerable": sum(not row["answerable"] for row in subset),
                        "min_tokens": min(lengths), "max_tokens": max(lengths),
                        "mean_tokens": sum(lengths) / len(lengths),
                        "candidate_counts": dict(Counter(candidates)),
                        "tiers": dict(Counter(row["tier"] for row in subset))}
    for split, subset in splits.items():
        destination = V1 / ("test.sealed.jsonl" if split == "test" else f"{split}.jsonl")
        with destination.open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    (V1 / "tokenizer.json").write_text(json.dumps({"schema_version": 1, "type": "train_only_byte_bpe",
                                                  "target_size": tok.target_size, "training_ids": [r["id"] for r in splits["train"]],
                                                  "merges": tok.merges}, separators=(",", ":")) + "\n")
    names = ("config.json", "train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json")
    manifest = {"schema_version": 1, "source": "original synthetic technical templates; no external text",
                "stats": stats, "split_families": config["families_by_split"],
                "entity_pool_sha256": {split: hashlib.sha256("\n".join(pool).encode()).hexdigest()
                                       for split, pool in entity_pools(config["seed"]).items()},
                "tokenizer_fit_split": "train", "test_sealed_for_selection": True,
                "file_sha256": {name: sha(V1 / name) for name in names}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stats


def verify():
    manifest = json.loads((V1 / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha(V1 / name) != expected:
            raise ValueError(f"Frozen corpus file changed: {name}")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.build == args.verify:
        parser.error("Choose one mode")
    print(json.dumps(build() if args.build else verify(), indent=2))
