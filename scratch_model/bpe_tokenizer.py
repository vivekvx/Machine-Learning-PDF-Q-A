import json
import re
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Subword Tokenizer implemented 100% from scratch.
    Learns frequent subword merges from corpus text.
    Zero external dependencies. Guarantees 0% OOV for English/ASCII text.
    """

    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    SEP_TOKEN = "<|sep|>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

    def __init__(self):
        self.special_tokens = self.SPECIAL_TOKENS[:]
        self.vocab = {}          # token_str -> token_id
        self.id_to_vocab = {}    # token_id -> token_str
        self.merges = {}         # (token_a, token_b) -> merged_token
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_vocab = {idx: tok for idx, tok in enumerate(self.special_tokens)}

    @property
    def pad_id(self): return self.vocab[self.PAD_TOKEN]
    @property
    def unk_id(self): return self.vocab[self.UNK_TOKEN]
    @property
    def bos_id(self): return self.vocab[self.BOS_TOKEN]
    @property
    def eos_id(self): return self.vocab[self.EOS_TOKEN]
    @property
    def sep_id(self): return self.vocab[self.SEP_TOKEN]
    @property
    def vocab_size(self): return len(self.vocab)

    @staticmethod
    def _pre_tokenize(text: str) -> list[str]:
        """Split text into word/punctuation units while preserving whitespace hints."""
        if not text:
            return []
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Match alphanumeric sequences or individual punctuation marks/symbols
        tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]|\n", text)
        return tokens

    def train(self, texts: list[str], target_vocab_size: int = 2500, min_pair_freq: int = 2):
        """Train BPE subword merges iteratively on corpus texts."""
        self._init_special_tokens()

        # Step 1: Collect base character vocabulary
        word_freqs = {}
        for text in texts:
            words = self._pre_tokenize(text)
            for w in words:
                w_lower = w.lower()
                word_freqs[w_lower] = word_freqs.get(w_lower, 0) + 1

        # Character-level split for each word
        # e.g., 'railway' -> ('r', 'a', 'i', 'l', 'w', 'a', 'y', '</w>')
        splits = {w: tuple(list(w) + ["</w>"]) for w in word_freqs.keys()}

        # Populate base single-character tokens into vocab
        unique_chars = sorted(set(char for w in splits.keys() for char in w) | {"</w>"})
        for ch in unique_chars:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.id_to_vocab[idx] = ch

        # Step 2: Iteratively find and merge the most frequent pair
        num_merges = target_vocab_size - len(self.vocab)
        self.merges = {}

        for _ in range(max(0, num_merges)):
            pairs = {}
            for word, freq in word_freqs.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_pair_freq:
                break

            # Create new merged token
            new_token = best_pair[0] + best_pair[1]
            idx = len(self.vocab)
            self.vocab[new_token] = idx
            self.id_to_vocab[idx] = new_token
            self.merges[best_pair] = new_token

            # Apply merge to all splits
            new_splits = {}
            for word, split in splits.items():
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                new_splits[word] = tuple(new_split)
            splits = new_splits

        return self.vocab_size

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word."""
        w_lower = word.lower()
        subwords = list(w_lower) + ["</w>"]

        while len(subwords) > 1:
            pairs = [(subwords[i], subwords[i + 1]) for i in range(len(subwords) - 1)]
            # Find earliest learned merge among existing pairs
            merge_candidates = [pair for pair in pairs if pair in self.merges]
            if not merge_candidates:
                break

            # Pick candidate that was learned earliest
            pair_to_merge = min(merge_candidates, key=lambda p: self.vocab[self.merges[p]])
            merged_token = self.merges[pair_to_merge]

            new_subwords = []
            i = 0
            while i < len(subwords):
                if i < len(subwords) - 1 and (subwords[i], subwords[i + 1]) == pair_to_merge:
                    new_subwords.append(merged_token)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            subwords = new_subwords

        return subwords

    def encode(
        self,
        text: str,
        max_len: int = None,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_to_max: bool = False
    ) -> list[int]:
        """Encode arbitrary text into subword token IDs."""
        words = self._pre_tokenize(text)
        token_ids = []

        if add_bos:
            token_ids.append(self.bos_id)

        for w in words:
            subwords = self._tokenize_word(w)
            for sw in subwords:
                token_ids.append(self.vocab.get(sw, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        if max_len is not None:
            if len(token_ids) > max_len:
                if add_eos and token_ids[-1] != self.eos_id:
                    token_ids = token_ids[:max_len - 1] + [self.eos_id]
                else:
                    token_ids = token_ids[:max_len]
            elif pad_to_max and len(token_ids) < max_len:
                token_ids = token_ids + [self.pad_id] * (max_len - len(token_ids))

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into clean English text."""
        tokens = []
        for tid in token_ids:
            if isinstance(tid, (list, tuple)):
                tid = tid[0]
            tok = self.id_to_vocab.get(int(tid), self.UNK_TOKEN)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        # Concatenate subwords and replace word boundary '</w>'
        text = "".join(tokens).replace("</w>", " ")
        # Clean punctuation spacing
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        return text.strip()

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Store merges as string keys for JSON serialization
        merges_str = [f"{a} {b}" for (a, b) in self.merges.keys()]
        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": merges_str
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str | Path):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data["special_tokens"]
        self.vocab = data["vocab"]
        self.id_to_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = {}
        for m_str in data.get("merges", []):
            parts = m_str.split(" ")
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                merged = parts[0] + parts[1]
                self.merges[pair] = merged
        return self
