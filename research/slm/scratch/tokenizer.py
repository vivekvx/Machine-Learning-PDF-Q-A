"""Deterministic byte-level BPE trained only on local synthetic training text.

No pretrained vocabulary, merges, or external tokenizer library is used.
"""
import json
from collections import Counter
from pathlib import Path

SPECIAL = {"pad": 0, "bos": 1, "eos": 2, "sep": 3}
BYTE_OFFSET = 4
BASE_VOCAB_SIZE = BYTE_OFFSET + 256


class ByteBPE:
    def __init__(self, merges=None, target_size=2000):
        self.merges = [tuple(item) for item in (merges or [])]
        self.target_size = target_size
        self.pair_to_new = {(a, b): (index, new) for index, (a, b, new) in enumerate(self.merges)}
        self.token_bytes = {BYTE_OFFSET + b: bytes([b]) for b in range(256)}
        for a, b, new in self.merges:
            if a not in self.token_bytes or b not in self.token_bytes or new in self.token_bytes:
                raise ValueError("Invalid BPE merge ordering")
            self.token_bytes[new] = self.token_bytes[a] + self.token_bytes[b]

    @property
    def vocab_size(self):
        return BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(cls, texts, target_size=2000):
        if target_size < BASE_VOCAB_SIZE:
            raise ValueError("Target vocabulary must contain all 256 byte values and four special IDs")
        docs = [[BYTE_OFFSET + value for value in text.encode("utf-8")]
                for text in texts]
        if not docs or not all(docs):
            raise ValueError("Nonempty training text required")
        merges = []
        while BASE_VOCAB_SIZE + len(merges) < target_size:
            pairs = Counter(pair for doc in docs for pair in zip(doc, doc[1:]))
            if not pairs:
                break
            pair = min(pairs, key=lambda item: (-pairs[item], item))
            new_id = BASE_VOCAB_SIZE + len(merges)
            merged = []
            for doc in docs:
                out = []
                i = 0
                while i < len(doc):
                    if i + 1 < len(doc) and (doc[i], doc[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(doc[i])
                        i += 1
                merged.append(out)
            docs = merged
            merges.append((pair[0], pair[1], new_id))
        return cls(merges, target_size)

    def encode(self, text, *, bos=False, eos=False):
        ids = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        # Apply merges in learned order; ranks are the training iteration numbers.
        for a, b, new_id in self.merges:
            if len(ids) < 2:
                break
            out = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ([SPECIAL["bos"]] if bos else []) + ids + ([SPECIAL["eos"]] if eos else [])

    def decode(self, ids, *, skip_special=True):
        parts = []
        for token in ids:
            token = int(token)
            if token in SPECIAL.values():
                if skip_special:
                    continue
                raise ValueError("Cannot decode special token as ordinary text")
            if token not in self.token_bytes:
                raise ValueError(f"Unknown token ID {token}")
            parts.append(self.token_bytes[token])
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"schema_version": 1, "type": "byte_bpe_from_synthetic_train_only",
                "target_size": self.target_size, "vocab_size": self.vocab_size,
                "special_ids": SPECIAL, "merges": self.merges}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["special_ids"] != SPECIAL or data["type"] != "byte_bpe_from_synthetic_train_only":
            raise ValueError("Unexpected tokenizer identity")
        result = cls(data["merges"], data["target_size"])
        if result.vocab_size != data["vocab_size"]:
            raise ValueError("Tokenizer vocabulary count mismatch")
        return result
