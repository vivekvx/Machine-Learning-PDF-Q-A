import json
import re
from pathlib import Path


class WordTokenizer:
    """
    Pure Python Word-Level Tokenizer for Question Answering.
    Handles words, punctuation, numbers, and special tokens without external libraries.
    """

    PAD_TOKEN = "<pad>"  # Index 0: Padding for batch alignment
    UNK_TOKEN = "<unk>"  # Index 1: Out-of-vocabulary words
    SOS_TOKEN = "<sos>"  # Index 2: Start of sequence
    EOS_TOKEN = "<eos>"  # Index 3: End of sequence
    SEP_TOKEN = "<sep>"  # Index 4: Separator between Context and Question

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.word2idx = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.idx2word = {idx: token for idx, token in enumerate(self.SPECIAL_TOKENS)}

    @property
    def pad_id(self):
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self):
        return self.word2idx[self.UNK_TOKEN]

    @property
    def sos_id(self):
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_id(self):
        return self.word2idx[self.EOS_TOKEN]

    @property
    def sep_id(self):
        return self.word2idx[self.SEP_TOKEN]

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into lowercase words and punctuation symbols."""
        if not text:
            return []
        # Match words/numbers or individual punctuation marks
        return [tok.lower() for tok in re.findall(r"\w+|[^\w\s]", text)]

    def build_vocab(self, texts: list[str], max_vocab: int = 3000, min_freq: int = 1):
        """Construct vocabulary from a collection of text documents."""
        self._init_special_tokens()
        freq = {}

        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

        # Sort by frequency descending
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        for token, count in sorted_tokens:
            if count >= min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                if len(self.word2idx) >= max_vocab:
                    break

        return self.vocab_size

    def encode(
        self,
        text: str,
        max_len: int = None,
        add_sos: bool = False,
        add_eos: bool = False,
        pad_to_max: bool = False
    ) -> list[int]:
        """Convert a text string into a list of token IDs."""
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, self.unk_id) for t in tokens]

        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_len is not None:
            if len(ids) > max_len:
                if add_eos and ids[-1] != self.eos_id:
                    ids = ids[:max_len - 1] + [self.eos_id]
                else:
                    ids = ids[:max_len]
            elif pad_to_max and len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))

        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back into a readable English string."""
        words = []
        for idx in token_ids:
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            token = self.idx2word.get(int(idx), self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            words.append(token)

        # Re-attach punctuation naturally
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        text = re.sub(r"\s+(['’])\s+", r"\1", text)
        return text.strip()

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "vocab_size": self.vocab_size
            }, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str | Path):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        return self
