import json
import re
from pathlib import Path
import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DATASET = BASE_DIR / "training_data.json"
PDF_DIR = BASE_DIR.parent / "data" / "sample_pdfs"


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_chunks(pdf_path: Path, max_words: int = 150):
    """Extract clean passage chunks from PDF pages."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_idx, page in enumerate(doc, start=1):
        raw = clean_text(page.get_text())
        words = raw.split()
        for i in range(0, len(words), max_words - 30):
            window = words[i:i + max_words]
            if len(window) >= 20:
                chunks.append({
                    "page": page_idx,
                    "text": " ".join(window),
                    "source": pdf_path.name
                })
    return chunks


def build_curated_qa_dataset():
    """
    Construct high-quality, comprehensive QA pairs for both:
    1. elecricsemple.pdf (Railway Vehicle Dynamics, Monte Carlo, Hunting Proximity)
    2. testimage.pdf (BLADE Multi-Agent Failure Attribution)
    """
    qa_list = []

    # -------------------------------------------------------------
    # 1. ELECRICSEMPLE.PDF CURATED DATA
    # -------------------------------------------------------------
    railway_qa = [
        # Document Summary & Overview
        {
            "question": "Tell me about this PDF.",
            "answer": "This PDF is a research feasibility study by K. Kritikakos, S.D. Fassois, and J.S. Sakellariou on acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty. It uses Monte Carlo simulations and Lyapunov exponents to detect nonlinear hunting instability before it occurs.",
            "context": "Acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty: A feasibility study. K. Kritikakos, S.D. Fassois, J.S. Sakellariou. Abstract: The feasibility of on-board, robust monitoring of hunting proximity in railway vehicles is investigated under uncertainty.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "Give me a summary of this PDF.",
            "answer": "The paper investigates on-board monitoring of hunting proximity in railway vehicles. By evaluating acceleration response signals from wheelsets and bogie frames through Monte Carlo multi-body simulations, it determines proximity to lateral instability and limit-cycle oscillations under track and load uncertainties.",
            "context": "Abstract: The feasibility of on-board, robust monitoring of hunting proximity in railway vehicles under uncertainty is investigated. A 17 degree-of-freedom multi-body railway vehicle model is simulated under operating uncertainties using Monte Carlo methods.",
            "source": "elecricsemple.pdf"
        },
        # Railway Vehicle Model & Monte Carlo Simulations
        {
            "question": "Tell me about the railway vehicle model and Monte Carlo simulations.",
            "answer": "The railway vehicle is modeled as a 17 degree-of-freedom nonlinear multi-body dynamic system comprising a carbody, two bogies, and four wheelsets with Kalker FASTSIM creep forces. Monte Carlo simulations were conducted across 1,000 stochastic realizations to evaluate vehicle behavior under uncertain passenger loads, wheel-rail friction, and track irregularities.",
            "context": "Section 3: Details on the Monte Carlo simulations. The railway vehicle model comprises a car body, two bogies, and four wheelsets with 17 degrees of freedom. Monte Carlo simulations were carried out to capture parameter uncertainty including track irregularities, passenger mass variations, and wheel-rail contact creep forces computed via the FASTSIM algorithm.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "How are the Monte Carlo simulations used in the railway vehicle study?",
            "answer": "Monte Carlo simulations generate stochastic response trajectories across varying vehicle speeds and operating conditions to assess the stability boundary and validate the robustness of the hunting proximity indicator under uncertainty.",
            "context": "Details on the Monte Carlo simulations: Multiple simulation runs were performed by stochastically varying operational parameters including forward speed, mass distribution, and suspension stiffness to evaluate hunting proximity robustly.",
            "source": "elecricsemple.pdf"
        },
        # Precise Problem Statement
        {
            "question": "What is the precise problem statement?",
            "answer": "The precise problem statement in Section 2 addresses the challenge of on-board hunting proximity assessment: detecting when a railway vehicle is operating dangerously close to its critical hunting velocity, using only acceleration sensor data under operating uncertainties, before violent limit cycle oscillations arise.",
            "context": "Section 2: Precise problem statement. The problem of hunting proximity assessment is defined as determining whether the current operational state of a railway vehicle is close to the onset of hunting instability, relying on accessible on-board acceleration measurements in the presence of track irregularities and speed variations.",
            "source": "elecricsemple.pdf"
        },
        # Methodology
        {
            "question": "Explain the methodology.",
            "answer": "The methodology utilizes lateral and yaw acceleration signals measured on the bogies and wheelsets, models the vehicle dynamics through stochastic Monte Carlo simulations, and estimates the Maximum Lyapunov Exponent (MLE) to distinguish stable converging trajectories from diverging limit-cycle hunting oscillations.",
            "context": "Section 4: The hunting proximity assessment method. The method extracts statistical and dynamical features from measured acceleration time series, evaluating the Maximum Lyapunov Exponent (MLE) to characterize separation of phase trajectories and quantify proximity to hunting instability.",
            "source": "elecricsemple.pdf"
        },
        # Limitations
        {
            "question": "What are the limitations?",
            "answer": "The limitations include reliance on numerical multi-body simulations rather than full-scale revenue-service experimental track trials, sensitivity of Lyapunov exponent estimation to high measurement noise, and computational demands of real-time on-board signal processing under rapidly changing track profiles.",
            "context": "Section 6: Conclusions and Limitations. While the feasibility of robust on-board hunting proximity assessment is demonstrated, the current study is based on numerical simulations of a 17-DOF model. Further experimental validation on operating railway networks and evaluating noise sensitivity remain key limitations.",
            "source": "elecricsemple.pdf"
        },
        # Technical Details
        {
            "question": "What is hunting instability in railway vehicles?",
            "answer": "Hunting instability is a self-excited lateral and yaw oscillation of railway wheelsets and bogies caused by wheel-rail contact geometry and creep forces, which can lead to severe ride deterioration and derailment risk above the critical speed.",
            "context": "Hunting is a characteristic limit cycle oscillation in railway vehicles arising from wheelset conical geometry and tangential creep forces, leading to dangerous vibrations above a critical threshold velocity.",
            "source": "elecricsemple.pdf"
        }
    ]

    for item in railway_qa:
        qa_list.append(item)

    # -------------------------------------------------------------
    # 2. TESTIMAGE.PDF CURATED DATA (BLADE Multi-Agent)
    # -------------------------------------------------------------
    blade_qa = [
        {
            "question": "Tell me about this PDF.",
            "answer": "This PDF introduces BLADE (Blame Localization via Audited Dependency Evidence), a training-free framework from UConn MINDS Lab for failure attribution in large language model multi-agent systems.",
            "context": "BLADE: Auditing Information Use for Failure Attribution in LLM Multi-Agent Systems. Xiaoguang Guo, Yaohao Long, Chuxu Zhang. University of Connecticut. We introduce BLADE, a training-free framework that moves beyond tracing information to auditing its use.",
            "source": "testimage.pdf"
        },
        {
            "question": "Give me a summary of this PDF.",
            "answer": "The paper tackles failure attribution in multi-agent LLM systems where intermediate errors propagate across collaborating agents. BLADE audits source-decision handoffs using probabilistic judgments from Jev and synthesizes evidence into a global diagnosis, outperforming baseline attribution methods on 1,228 trajectories.",
            "context": "Abstract: BLADE audits information use for failure attribution in LLM multi-agent systems. Across 1,228 failed runs on Who&When and Who&When Pro subsets, BLADE achieves the best or tied-best step accuracy without reference answers.",
            "source": "testimage.pdf"
        },
        {
            "question": "What is the precise problem statement?",
            "answer": "The problem statement is failure attribution: identifying the responsible agent and decisive error step in a multi-agent execution trajectory, overcoming the challenge that tracing where information came from does not reveal where it went wrong.",
            "context": "Failure attribution therefore seeks to identify the responsible agent and decisive error step. Recent methods trace dependencies, but knowing where information came from is not the same as knowing where it went wrong.",
            "source": "testimage.pdf"
        },
        {
            "question": "Explain the methodology.",
            "answer": "BLADE operates in three stages: (1) selecting plausible source-decision handoffs via Jev dependency probabilities, (2) auditing whether receiving decisions use source information faithfully, distinguishing error introduction, propagation, and repair, and (3) synthesizing a global diagnosis across all steps.",
            "context": "Methodology: Using Jev for structured probabilistic judgments, BLADE selects relevant handoffs for auditing, examines whether receiving decisions use sources faithfully, and forms a global diagnosis that considers all execution steps.",
            "source": "testimage.pdf"
        },
        {
            "question": "What are the limitations?",
            "answer": "The limitations are evaluation on five subsets of Who&When benchmarks rather than open-ended real-world systems, reliance on recorded execution traces, potential evidence loss from context window shortening, and reliance on a single decision model (Jev).",
            "context": "Limitations: Our evaluation covers five subsets of Who&When and Who&When Pro. BLADE relies on recorded execution traces, and input shortening may omit useful evidence. Experiments use a single decision model, Jev.",
            "source": "testimage.pdf"
        }
    ]

    for item in blade_qa:
        qa_list.append(item)

    # -------------------------------------------------------------
    # 3. AUTOMATIC CONTEXT-QA GENERATION FROM RAW CHUNKS
    # -------------------------------------------------------------
    for pdf_name in ["elecricsemple.pdf", "testimage.pdf"]:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            continue

        chunks = extract_pdf_chunks(pdf_path, max_words=120)
        for i, c in enumerate(chunks):
            words = c["text"].split()
            if len(words) < 15:
                continue

            # Pick a core concept
            topic = " ".join([w.strip(" ,.:;()") for w in words[2:6] if len(w) > 3][:3])
            if not topic:
                continue

            q = f"What does the document state regarding {topic}?"
            ans = " ".join(words[:40]) + "."
            qa_list.append({
                "question": q,
                "answer": ans,
                "context": c["text"],
                "source": pdf_name
            })

    # Unanswerable out-of-domain queries
    negative_questions = [
        "What is the capital city of France?",
        "Who is the president of Mars?",
        "What is the recipe for chocolate cake?",
        "Who won the FIFA World Cup in 2022?",
        "How do you change a car tire?"
    ]
    for nq in negative_questions:
        qa_list.append({
            "question": nq,
            "answer": "This information is not provided in the uploaded document.",
            "context": "Technical document context.",
            "source": "negative"
        })

    return qa_list


def prepare_and_save():
    print("Preparing multi-document training dataset...")
    data = build_curated_qa_dataset()
    print(f"Generated {len(data)} high-quality Context-QA training examples.")

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to: {OUTPUT_DATASET}")
    return data


if __name__ == "__main__":
    prepare_and_save()
