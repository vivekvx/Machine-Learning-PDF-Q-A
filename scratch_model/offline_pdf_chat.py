import streamlit as st
import requests
import fitz  # PyMuPDF

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"

st.set_page_config(
    page_title="Offline PDF AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Offline PDF Question Answering")
st.write("Upload a PDF and ask questions about its contents.")

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    try:
        document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()
            if text:
                pages.append(f"[Page {page_number}]\n{text}")

        st.session_state.pdf_text = "\n\n".join(pages)

        if st.session_state.pdf_text:
            st.success(f"PDF loaded successfully! Pages: {len(pages)}")
        else:
            st.warning(
                "No selectable text found. This PDF may be scanned; OCR is needed."
            )

    except Exception as error:
        st.error(f"Could not read PDF: {error}")

if st.session_state.pdf_text:
    question = st.chat_input("Ask a question about your PDF...")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Keep the context small enough for the local model.
        pdf_context = st.session_state.pdf_text[:12000]

        prompt = f"""
You are a PDF question-answering assistant.
Answer using ONLY the information in the PDF text below.
If the answer is not present in the provided text, say:
"The PDF does not provide enough information to answer this question."
Do not invent facts.

PDF TEXT:
{pdf_context}

QUESTION:
{question}
"""

        with st.chat_message("assistant"):
            with st.spinner("Reading the PDF and generating an answer..."):
                try:
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": MODEL,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "Answer questions strictly using the "
                                        "provided PDF context."
                                    )
                                },
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "options": {"temperature": 0.1}
                        },
                        timeout=180
                    )

                    response.raise_for_status()
                    answer = response.json()["message"]["content"]

                except requests.exceptions.ConnectionError:
                    answer = (
                        "Cannot connect to Ollama. Open another terminal "
                        "and run: ollama serve"
                    )
                except Exception as error:
                    answer = f"Error generating answer: {error}"

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("Upload a text-based PDF to begin chatting.")

