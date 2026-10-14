"""Deterministic byte BPE trained from scratch on v3 training documents only.

Whitespace and non-whitespace runs are independent pre-tokens. This keeps
training bounded on the larger corpus while retaining exact UTF-8 roundtrips.
"""
import re
from collections import Counter
from functools import lru_cache

SPECIAL = {"pad": 0, "bos": 1, "eos": 2, "sep": 3}
BASE = 260
PIECES = re.compile(r"\s+|[^\s]+")


class V3ByteBPE:
    def __init__(self, merges, target_size=1024):
        self.merges = [tuple(int(x) for x in merge) for merge in merges]
        self.target_size = target_size
        self.token_bytes = {i + 4: bytes([i]) for i in range(256)}
        for left, right, new in self.merges:
            if left not in self.token_bytes or right not in self.token_bytes or new in self.token_bytes:
                raise ValueError("Invalid merge ordering")
            self.token_bytes[new] = self.token_bytes[left] + self.token_bytes[right]

    @property
    def vocab_size(self):
        return BASE + len(self.merges)

    @classmethod
    def train(cls, texts, target_size=1024):
        if target_size < BASE:
            raise ValueError("Byte fallback vocabulary is required")
        frequencies = Counter(piece for text in texts for piece in PIECES.findall(text))
        if not frequencies:
            raise ValueError("Training text is empty")
        sequences = {piece: [byte + 4 for byte in piece.encode("utf-8")] for piece in frequencies}
        merges = []
        for new in range(BASE, target_size):
            pairs = Counter()
            for piece, ids in sequences.items():
                pairs.update({pair: count * frequencies[piece]
                              for pair, count in Counter(zip(ids, ids[1:])).items()})
            if not pairs:
                break
            pair = min(pairs, key=lambda value: (-pairs[value], value))
            left, right = pair
            for piece, ids in sequences.items():
                if left not in ids or right not in ids:
                    continue
                out = []
                index = 0
                while index < len(ids):
                    if index + 1 < len(ids) and ids[index] == left and ids[index + 1] == right:
                        out.append(new)
                        index += 2
                    else:
                        out.append(ids[index])
                        index += 1
                sequences[piece] = out
            merges.append((left, right, new))
        return cls(merges, target_size)

    @lru_cache(maxsize=100000)
    def _encode_piece(self, piece):
        ids = [byte + 4 for byte in piece.encode("utf-8")]
        for left, right, new in self.merges:
            if len(ids) < 2:
                break
            out = []
            index = 0
            while index < len(ids):
                if index + 1 < len(ids) and ids[index] == left and ids[index + 1] == right:
                    out.append(new)
                    index += 2
                else:
                    out.append(ids[index])
                    index += 1
            ids = out
        return tuple(ids)

    def encode(self, text, *, bos=False, eos=False):
        ids = [SPECIAL["bos"]] if bos else []
        for piece in PIECES.findall(text):
            ids.extend(self._encode_piece(piece))
        if eos:
            ids.append(SPECIAL["eos"])
        return ids

    def decode(self, ids):
        return b"".join(self.token_bytes[int(token)] for token in ids if int(token) not in SPECIAL.values()).decode("utf-8", errors="strict")
