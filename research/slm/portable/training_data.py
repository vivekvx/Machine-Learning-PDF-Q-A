"""Shared Qwen chat serialization and assistant-only target construction."""
import json
from collections.abc import Mapping
from pathlib import Path


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def local_checkpoint_complete(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text()).get("weight_map", {})
        return bool(mapping) and all((path / filename).exists() for filename in set(mapping.values()))
    return (path / "model.safetensors").exists()


def _ids(tokenizer, messages, **kwargs):
    result = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    return result["input_ids"] if isinstance(result, Mapping) else result


def tokenized_row(row, tokenizer, maximum_sequence_tokens=512):
    messages = row["messages"]
    if messages[-1]["role"] != "assistant" or messages[-1]["content"] != row["target_answer"]:
        raise ValueError(f"Assistant target mismatch: {row['question_id']}")
    prefix = _ids(tokenizer, messages[:-1], add_generation_prompt=True)
    full = _ids(tokenizer, messages)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix) + 1:
        raise ValueError(f"Qwen template prefix mismatch: {row['question_id']}")
    if len(full) != row["full_tokens"] or len(prefix) != row["prompt_tokens"]:
        raise ValueError(f"Tokenizer/template drift: {row['question_id']}")
    if len(full) > maximum_sequence_tokens:
        raise ValueError(f"Overlong training example: {row['question_id']}")
    if row["answerability"] == "unanswerable" and row["target_answer"] != "UNANSWERABLE":
        raise ValueError(f"Wrong unanswerable target: {row['question_id']}")
    labels = [-100] * len(prefix) + full[len(prefix):]
    if all(label == -100 for label in labels):
        raise ValueError(f"No supervised assistant tokens: {row['question_id']}")
    return full, len(prefix), labels


class MLXCompletionDataset:
    def __init__(self, path, tokenizer, maximum_sequence_tokens=512):
        self.data = list(rows(path))
        self.tokenizer = tokenizer
        self.maximum_sequence_tokens = maximum_sequence_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def process(self, row):
        full, offset, _ = tokenized_row(row, self.tokenizer, self.maximum_sequence_tokens)
        return full, offset


class PeftCompletionDataset:
    def __init__(self, path, tokenizer, maximum_sequence_tokens=512):
        self.data = []
        for row in rows(path):
            full, _, labels = tokenized_row(row, tokenizer, maximum_sequence_tokens)
            self.data.append({"input_ids": full, "labels": labels,
                              "attention_mask": [1] * len(full)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def padded_collator(features, pad_token_id):
    import torch

    length = max(len(x["input_ids"]) for x in features)
    return {"input_ids": torch.tensor([x["input_ids"] + [pad_token_id] * (length - len(x["input_ids"]))
                                       for x in features], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] + [0] * (length - len(x["attention_mask"]))
                                            for x in features], dtype=torch.long),
            "labels": torch.tensor([x["labels"] + [-100] * (length - len(x["labels"]))
                                    for x in features], dtype=torch.long)}
