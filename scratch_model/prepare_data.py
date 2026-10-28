import json
import re
import sys
from pathlib import Path

# Import your existing PDF loader
from src.pdf_loader import load_pdf

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "training_data.json"


def split_into_sentences(text):
    """Split extracted PDF text into simple sentences."""
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 5
    ]


def create_question(sentence):
    """Create a basic question from a sentence."""
    words = sentence.split()

    if len(words) < 5:
        return None

    # Use the first few words as a simple topic prompt.
    topic = " ".join(words[:5]).strip(" ,.:;")

    return f"What does the document say about {topic}?"


def prepare_training_data(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path

    print(f"\nReading PDF: {pdf_path.name}")

    pages = load_pdf(pdf_path)
    examples = []

    for page in pages:
        sentences = split_into_sentences(page.page_content)

        for sentence in sentences:
            question = create_question(sentence)

            if question is None:
                continue

            examples.append({
                "question": question,
                "answer": sentence,
                "source": page.source,
                "page": page.page_number
            })

    if not examples:
        raise ValueError(
            "No suitable sentences found. Try a text-based PDF."
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(examples, file, indent=2, ensure_ascii=False)

    print(f"\nPDF pages processed: {len(pages)}")
    print(f"Training examples created: {len(examples)}")
    print(f"Dataset saved to: {OUTPUT_FILE}")

    print("\nSample examples:")

    for example in examples[:3]:
        print("\nQuestion:", example["question"])
        print("Answer:", example["answer"])
        print("Page:", example["page"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python -m scratch_model.prepare_data "path\\to\\file.pdf"'
        )
        sys.exit(1)

    prepare_training_data(sys.argv[1])

