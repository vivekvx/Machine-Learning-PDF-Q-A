"""
GPT-2 Fine-Tuned Model Inference Module
=========================================
Provides the generate_answer() function used by the Streamlit chat app.
Loads the fine-tuned GPT-2 model and generates structured answers from PDF context.
"""

import re
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2QAModel:
    """
    Wrapper for the fine-tuned GPT-2 model that generates answers
    conditioned on retrieved PDF context.

    Model: GPT-2 Small (117M params) fine-tuned on our custom QA dataset.
    The pretrained GPT-2 weights provide English language understanding.
    The fine-tuning teaches the model our specific QA format and to
    extract/structure information from provided context.
    """

    def __init__(self, checkpoint_dir: str | Path = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parent / "checkpoint" / "gpt2_finetuned"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = None
        self.tokenizer = None
        self.meta = {}
        self._loaded = False

    def load(self):
        """Load the fine-tuned model and tokenizer."""
        import json

        if self._loaded:
            return self

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {self.checkpoint_dir}\n"
                f"Please run train_gpt2.py first."
            )

        print(f"Loading fine-tuned GPT-2 from {self.checkpoint_dir}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.checkpoint_dir))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(str(self.checkpoint_dir))
        self.model.to(self.device)
        self.model.eval()

        # Load training metadata
        meta_path = self.checkpoint_dir / "training_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.meta = json.load(f)

        self._loaded = True
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params:,} parameters on {self.device.upper()}")
        return self

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.3,
    ) -> tuple[str, float]:
        """
        Generate an answer given a question and retrieved PDF context.

        Args:
            question: The user's question
            context: Retrieved PDF text chunks
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens

        Returns:
            (answer_text, generation_time_seconds)
        """
        if not self._loaded:
            self.load()

        t0 = time.time()

        # Truncate context to fit within model's context window
        # GPT-2 max is 1024 tokens. We need room for question + answer.
        # Reserve ~350 tokens for answer, ~50 for question+formatting
        max_context_tokens = 600
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(context_ids) > max_context_tokens:
            context_ids = context_ids[:max_context_tokens]
            context = self.tokenizer.decode(context_ids, skip_special_tokens=True)

        # Build prompt in the same format used during training
        # For small causal models, providing instructional prefix steers structured output
        prompt = f"Context: {context}\n\nQuestion: {question.strip()}\n\nAnswer: Based on the document,"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Ensure we don't exceed GPT-2's max position embeddings
        if input_ids.shape[1] > 1024 - max_new_tokens:
            # Trim from the start of context
            excess = input_ids.shape[1] - (1024 - max_new_tokens)
            input_ids = input_ids[:, excess:]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),  # Avoid division by zero
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=4,  # Prevent 4-gram repetitions
            )

        # Extract only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        answer = "Based on the document, " + self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Post-processing: clean up the answer
        answer = self._clean_answer(answer)

        gen_time = time.time() - t0
        return answer, gen_time

    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer text."""
        # Remove any prompt leakage (Context:, Question:, etc.)
        stop_patterns = [
            r"\bContext:", r"\bQuestion:", r"\bAnswer:",
            r"<\|endoftext\|>", r"<\|pad\|>", r"<\|sep\|>",
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                answer = answer[:match.start()].strip()

        # Remove trailing incomplete sentences (if answer ends mid-sentence)
        if answer and answer[-1] not in ".!?\n:":
            # Find the last sentence-ending punctuation
            last_period = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"), answer.rfind("\n"))
            if last_period > len(answer) * 0.5:  # Only trim if we keep at least half
                answer = answer[:last_period + 1]

        # Clean up extra whitespace
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        answer = answer.strip()

        return answer

    def get_model_info(self) -> dict:
        """Return model metadata for display in the UI."""
        if not self._loaded:
            self.load()

        return {
            "device": self.device,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_len": 1024,
            "base_model": self.meta.get("base_model", "gpt2"),
            "val_loss": self.meta.get("best_val_loss", 0),
            "train_loss": self.meta.get("best_train_loss", 0),
            "training_approach": self.meta.get("training_approach", "Fine-tuned GPT-2"),
            "pretrained_weights": self.meta.get("pretrained_weights", "GPT-2 Small (OpenAI)"),
            "training_examples": self.meta.get("training_examples", 0),
        }


# Convenience function for direct usage
def load_model(checkpoint_dir=None):
    """Load and return the fine-tuned GPT-2 QA model."""
    model = GPT2QAModel(checkpoint_dir)
    model.load()
    return model


if __name__ == "__main__":
    # Quick test
    qa_model = load_model()

    test_context = (
        "Railway hunting is a coupled lateral and yaw limit cycle oscillation "
        "of the vehicle, physically limited by the wheel flanges. It occurs at "
        "a traveling speed vh, referred to as the hunting speed, which depends "
        "on the specific track/vehicle conditions."
    )
    test_question = "What is hunting instability?"

    answer, gen_time = qa_model.generate_answer(test_question, test_context)
    print(f"\nQuestion: {test_question}")
    print(f"Answer: {answer}")
    print(f"Time: {gen_time:.2f}s")

