"""
Enhanced Training Data Preparation for GPT-2 Fine-Tuning
=========================================================
Generates a comprehensive instruction-style QA dataset from PDF documents.
Includes curated high-quality pairs, auto-generated pairs, and negative examples.
"""

import json
import re
from pathlib import Path
import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DATASET = BASE_DIR / "training_data_v2.json"
PDF_DIR = BASE_DIR.parent / "data" / "sample_pdfs"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_chunks(pdf_path: Path, max_words: int = 200, overlap: int = 50):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_idx, page in enumerate(doc, start=1):
        raw = clean_text(page.get_text())
        words = raw.split()
        step = max(1, max_words - overlap)
        for i in range(0, len(words), step):
            window = words[i:i + max_words]
            if len(window) >= 30:
                chunks.append({
                    "page": page_idx,
                    "text": " ".join(window),
                    "source": pdf_path.name
                })
    return chunks


def build_curated_qa_dataset():
    qa_list = []

    # =================================================================
    # 1. ELECRICSEMPLE.PDF — Railway Vehicle Hunting Proximity Paper
    # =================================================================
    railway_qa = [
        # --- Document Overview ---
        {
            "question": "Tell me about this PDF.",
            "answer": "This PDF is a research feasibility study titled 'Acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty' by K. Kritikakos, S.D. Fassois, and J.S. Sakellariou from the University of Patras, Greece.\n\nThe study investigates the feasibility of detecting how close a railway vehicle is to hunting instability using on-board acceleration sensor data. The key contributions include:\n\n1. A novel Maximum Lyapunov Exponent (MLE) based method for assessing hunting proximity during normal vehicle operation.\n2. A grey nonlinear vehicle model with physics-informed structure for accurate MLE estimation.\n3. Extensive Monte Carlo simulations demonstrating 97.08% classification accuracy.\n4. Robustness assessment under varying suspension, wheel, and track conditions.",
            "context": "Acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty: A feasibility study. K. Kritikakos, S.D. Fassois, J.S. Sakellariou. University of Patras, Department of Mechanical Engineering & Aeronautics. Abstract: The feasibility of on-board, real-time proximity assessment to railway hunting initiation under uncertainty arising from vehicle/track and other operational conditions is assessed. A novel method that classifies hunting proximity into one of several prespecified speed proximity intervals is introduced, based on a grey nonlinear vehicle model with parameters estimated from on-board acceleration measurements and model-based estimation of the Maximum Lyapunov Exponent (MLE) as a hunting proximity measure. The method achieves 97.08% classification accuracy.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "What is this document about?",
            "answer": "This document is a research paper that investigates whether railway vehicles can detect their proximity to hunting instability in real-time using on-board acceleration sensors.\n\nThe researchers developed a method that:\n1. Uses a grey nonlinear model of the vehicle's lateral dynamics.\n2. Estimates the Maximum Lyapunov Exponent (MLE) from acceleration measurements.\n3. Classifies hunting proximity into predefined speed intervals.\n4. Achieves 97.08% accuracy across extensive Monte Carlo simulations.\n\nThe study addresses the critical safety challenge of detecting dangerous hunting oscillations before they occur, under uncertain operating conditions including varying track quality, passenger loads, and wheel-rail friction.",
            "context": "Acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty: A feasibility study. The feasibility of on-board, real-time proximity assessment to railway hunting initiation under uncertainty is assessed. A novel method classifies hunting proximity into prespecified speed proximity intervals using a grey nonlinear vehicle model and MLE estimation. Performance assessed via Monte Carlo simulations achieving 97.08% classification accuracy.",
            "source": "elecricsemple.pdf"
        },
        # --- Summary ---
        {
            "question": "Give me a summary of this PDF.",
            "answer": "This paper presents a feasibility study on robust on-board monitoring of hunting proximity in railway vehicles. Here is a comprehensive summary:\n\n1. Objective: Assess whether hunting proximity can be reliably detected during normal vehicle operation using accessible acceleration data.\n\n2. Approach: The authors develop a grey nonlinear vehicle model with a physics-informed structure, estimate its parameters from on-board acceleration measurements, and compute the Maximum Lyapunov Exponent (MLE) as a hunting proximity indicator.\n\n3. Simulation Framework: A 17-degree-of-freedom (later refined to 21-DOF) multi-body railway vehicle model is simulated in MATLAB. Monte Carlo simulations with 1,000 stochastic realizations capture uncertainties in track irregularities, passenger mass, and wheel-rail contact.\n\n4. Results: The method achieves 97.08% classification accuracy in identifying hunting proximity intervals, significantly outperforming alternative approaches.\n\n5. Robustness: The method demonstrates robustness to faulty suspensions and hollow-worn wheels, though some performance compromise is observed under worn-track conditions.\n\n6. Significance: This is the first study to examine the feasibility of assessing proximity to hunting onset during normal vehicle operation.",
            "context": "Abstract: The feasibility of on-board, robust monitoring of hunting proximity in railway vehicles under uncertainty is investigated. A 17 degree-of-freedom multi-body railway vehicle model is simulated under operating uncertainties using Monte Carlo methods. A novel MLE-based method achieves 97.08% classification accuracy. The method demonstrates robustness to faulty suspensions and hollow-worn wheels.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "Summarize the key findings of this paper.",
            "answer": "The key findings of this paper are:\n\n1. Feasibility Confirmed: On-board hunting proximity assessment using acceleration data is feasible and achieves excellent performance with 97.08% classification accuracy.\n\n2. MLE as Indicator: The Maximum Lyapunov Exponent (MLE) serves as an effective hunting proximity measure, evolving from negative values (stable) toward zero (limit cycle) as speed approaches the hunting threshold.\n\n3. Grey Model Advantage: The proposed grey nonlinear model with physics-informed structure overcomes limitations of existing data-driven MLE estimation methods for high-order systems like railway vehicles.\n\n4. Robustness: The method maintains reliable performance under:\n   - Varying track irregularities\n   - Different passenger mass distributions\n   - Faulty suspension components\n   - Hollow-worn wheels\n\n5. Limitation Under Worn Tracks: Performance compromise was observed under worn-track conditions, indicating an area for further research.\n\n6. Superiority: The proposed method significantly outperforms an alternative indirect data-driven MLE estimation approach.",
            "context": "The method achieves 97.08% classification accuracy. Robustness to faulty suspensions and hollow-worn wheels is demonstrated. Some performance compromise under worn-track conditions. The MLE evolves from negative values toward zero as speed approaches hunting speed. The grey model approach overcomes limitations of data-driven MLE methods for high-order systems.",
            "source": "elecricsemple.pdf"
        },
        # --- Problem Statement ---
        {
            "question": "What is the precise problem statement?",
            "answer": "The precise problem statement, as defined in Section 2 of the paper, is:\n\nGiven an N-sample-long acceleration signal y[k] (k = 1, ..., N), available at time instant N through s uni-axial on-board sensors while the vehicle travels at a constant speed v, properly classify the proximity of the current speed v to the unknown and condition-dependent hunting speed vh into one of d user-defined intervals.\n\nIn simpler terms, the problem is:\n1. The vehicle has acceleration sensors mounted on its wheelsets and bogies.\n2. These sensors continuously record lateral acceleration data.\n3. The system must determine how close the current speed is to the dangerous hunting speed.\n4. This assessment must work correctly regardless of varying track and vehicle conditions (robustness).\n5. The hunting speed vh is unknown and changes with operating conditions.\n6. The proximity is classified into 8 predefined intervals of 10 km/h each.\n\nThe key challenge is performing this assessment in real-time, on-board, before violent limit cycle oscillations actually occur.",
            "context": "Section 2: Precise problem statement. The problem of hunting proximity assessment (classification) may be stated as follows: Given a N-sample-long vector acceleration signal y[k], available through s uni-axial on-board sensors while the vehicle travels at constant speed v, properly classify the proximity of the current speed v to the (unknown and condition-dependent) hunting speed vh into one of d user-defined intervals. Eight hunting proximity intervals (d = 8) are considered.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "What problem does this research address?",
            "answer": "This research addresses the problem of detecting proximity to hunting instability in railway vehicles during normal operation.\n\nSpecifically:\n1. Hunting is a dangerous lateral and yaw oscillation that can cause derailment.\n2. It occurs above a critical speed called the hunting speed, which varies with track and vehicle conditions.\n3. Current methods can only detect hunting after it has already started.\n4. This research aims to predict how close the vehicle is to hunting before it occurs.\n5. The detection must be robust to uncertainties in track quality, passenger load, and wheel-rail friction.\n6. Only standard on-board acceleration sensors should be required.",
            "context": "The central question raised in this study is whether the instantaneous proximity to hunting onset can be robustly and reliably evaluated during vehicle operation under any operating conditions and prior to hunting occurrence. Hunting is a coupled lateral and yaw limit cycle oscillation of the vehicle that occurs above the hunting speed.",
            "source": "elecricsemple.pdf"
        },
        # --- Methodology ---
        {
            "question": "Explain the methodology.",
            "answer": "The methodology consists of several integrated stages:\n\n1. Vehicle Modeling: A grey nonlinear model of the vehicle's lateral dynamics is developed with a physics-informed structure. Unlike black-box models, this incorporates known physical relationships while keeping certain parameters unknown.\n\n2. Parameter Estimation: The unknown model parameters are estimated from on-board acceleration measurements using Linear Least Squares (LLS) criteria.\n\n3. MLE Computation: Using the Jacobian matrix of the estimated grey model, the Maximum Lyapunov Exponent (MLE) is computed. The MLE characterizes whether nearby trajectories converge (stable), maintain constant separation (limit cycle), or diverge (chaotic).\n\n4. Proximity Classification: The MLE value is used to classify the vehicle's current hunting proximity into one of 8 predefined speed intervals.\n\n5. Validation: The method is validated through extensive Monte Carlo simulations using a MATLAB-based 21-DOF multi-body vehicle model across four operating scenarios with varying suspension, wheel, and track conditions.\n\nThe key innovation is using a grey model approach that overcomes the accuracy limitations of purely data-driven MLE estimation methods for high-order dynamical systems like railway vehicles.",
            "context": "Section 4: The hunting proximity assessment method. The method uses a grey nonlinear vehicle model with physics-informed structure, estimates parameters via LLS from acceleration measurements, computes MLE using the model Jacobian, and classifies hunting proximity. Validated via Monte Carlo simulations with four operating scenarios.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "How does the proposed method work?",
            "answer": "The proposed method works through the following steps:\n\n1. Data Acquisition: Lateral acceleration signals are collected from uni-axial accelerometers mounted on the vehicle's wheelsets and bogie frames.\n\n2. Grey Model Estimation: A nonlinear model of the vehicle's lateral dynamics is fitted to the acceleration data. This model has a physics-informed structure (known form) but unknown parameters estimated from the sensor data.\n\n3. Jacobian Computation: The model is differentiated to obtain the Jacobian matrix, which describes how the system state evolves over time.\n\n4. MLE Calculation: The Maximum Lyapunov Exponent is computed from the Jacobian. A negative MLE indicates stable behavior, while an MLE approaching zero indicates proximity to hunting (limit cycle).\n\n5. Classification: The MLE value is mapped to one of 8 hunting proximity intervals, each spanning 10 km/h of speed difference from the hunting speed.\n\nThis approach avoids the limitations of direct MLE estimation methods (sensitive to noise) and black-box indirect methods (poor accuracy for high-order systems).",
            "context": "The method extracts acceleration signals, estimates a grey nonlinear model, computes the Jacobian, estimates MLE, and classifies hunting proximity. MLE evolves from negative values toward zero as speed approaches hunting speed.",
            "source": "elecricsemple.pdf"
        },
        # --- Railway Vehicle Model ---
        {
            "question": "Tell me about the railway vehicle model and Monte Carlo simulations.",
            "answer": "The railway vehicle model and Monte Carlo simulations are described as follows:\n\nVehicle Model:\n1. A 21-degree-of-freedom (DOF) multi-body simulation model of lateral dynamics developed in MATLAB.\n2. Consists of 7 rigid bodies: one car-body, two bogie frames, and four wheelsets.\n3. Each wheelset has 3 DOFs: lateral deviation, longitudinal deviation, and yaw angle.\n4. Bogie frames and car-body also have 3 DOFs each: lateral deviation, yaw angle, and roll angle.\n5. Primary suspension (conical rubber springs) connects wheelsets to bogies.\n6. Secondary suspension (air springs) connects bogies to the car-body.\n7. Wheel-rail creep forces are computed using Kalker's linear theory with FASTSIM saturation.\n8. Based on Athens Metro vehicle parameters.\n\nMonte Carlo Simulations:\n1. Performed across four operating scenarios (A-D) with varying conditions.\n2. Random alignment track irregularity profiles following ERRI B176 standard.\n3. Constant vehicle speeds randomly selected within hunting proximity intervals.\n4. 22 acceleration signals recorded from accelerometers per simulation.\n5. Used to assess robustness under uncertain passenger loads, track quality, and component conditions.",
            "context": "A 21-degree-of-freedom railway vehicle simulation model of lateral dynamics is developed in MATLAB. The model consists of 7 rigid bodies including a car-body, two bogie frames, and four wheelsets. Monte Carlo simulations performed under four operating scenarios with random track irregularity profiles. Wheel-rail creep forces computed via FASTSIM algorithm.",
            "source": "elecricsemple.pdf"
        },
        # --- Limitations ---
        {
            "question": "What are the limitations?",
            "answer": "The limitations of this study include:\n\n1. Simulation-Based Validation Only: The study relies entirely on numerical multi-body simulations rather than full-scale experimental validation on revenue-service railway tracks.\n\n2. Worn-Track Performance: The method shows compromised performance under worn-track conditions, indicating sensitivity to certain track degradation patterns.\n\n3. Noise Sensitivity: The Lyapunov exponent estimation may be sensitive to high levels of measurement noise from real-world accelerometer data.\n\n4. Computational Requirements: Real-time on-board signal processing and MLE computation under rapidly changing track profiles may pose computational challenges.\n\n5. Constant Speed Assumption: The current formulation assumes constant vehicle speed over the observation interval, which may not hold during acceleration or braking phases.\n\n6. Single Track Type: Simulations use tangent (straight) track only; curved track dynamics are not addressed.\n\n7. Model Simplifications: Vertical and pitch motions are neglected, and the model assumes full geometric symmetry about the longitudinal plane.",
            "context": "Section 6: Conclusions and Limitations. The study is based on numerical simulations of a multi-body model. Some performance compromise under worn-track conditions. Sensitivity of Lyapunov exponent estimation to measurement noise. Computational demands of real-time processing. Constant speed assumption over observation interval.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "What are the weaknesses of this study?",
            "answer": "The weaknesses of this study are:\n\n1. No Experimental Validation: All results are based on numerical simulations, not real-world tests on operating railways.\n\n2. Performance Degradation on Worn Tracks: Classification accuracy drops when track conditions deviate significantly from nominal.\n\n3. Noise Sensitivity: MLE estimation using acceleration data may produce false indications under high measurement noise.\n\n4. Limited Track Geometry: Only tangent (straight) tracks are considered; curved tracks introduce additional dynamics.\n\n5. Simplified Model Assumptions: Vertical dynamics, pitch motions, and asymmetric conditions are neglected.",
            "context": "Limitations include simulation-only validation, worn-track performance compromise, noise sensitivity of MLE estimation, constant speed assumption, and neglect of vertical and pitch motions.",
            "source": "elecricsemple.pdf"
        },
        # --- Advantages ---
        {
            "question": "What are the advantages of the proposed method?",
            "answer": "The advantages of the proposed method are:\n\n1. High Accuracy: Achieves 97.08% classification accuracy in identifying hunting proximity intervals, demonstrating reliable performance.\n\n2. Proactive Detection: Assesses proximity to hunting before dangerous oscillations occur, unlike existing methods that only detect hunting after onset.\n\n3. Uses Standard Sensors: Relies on accessible uni-axial accelerometers already commonly installed on railway vehicles, requiring no specialized equipment.\n\n4. Robustness: Maintains reliable performance under varying conditions including faulty suspensions and hollow-worn wheels.\n\n5. Physics-Informed Approach: The grey model structure incorporates physical knowledge of vehicle dynamics, overcoming limitations of purely data-driven methods for high-order systems.\n\n6. On-Board Capability: Designed for real-time on-board implementation using only locally measured acceleration data.\n\n7. Condition-Independent: Works across different and typically unknown hunting speeds caused by varying track and vehicle conditions.",
            "context": "The method achieves 97.08% classification accuracy. Uses accessible on-board acceleration measurements. Demonstrates robustness to faulty suspensions and hollow-worn wheels. Grey model overcomes data-driven MLE limitations. Designed for on-board real-time assessment.",
            "source": "elecricsemple.pdf"
        },
        # --- Technical Concepts ---
        {
            "question": "What is hunting instability in railway vehicles?",
            "answer": "Hunting instability is a phenomenon of critical importance in the railway industry. It can be described as follows:\n\n1. Definition: Hunting is a coupled lateral and yaw limit cycle (periodic) oscillation of the railway vehicle, physically limited by the wheel flanges.\n\n2. Cause: It fundamentally arises from the nonlinear nature of railway vehicle dynamics, mainly associated with wheel-rail contact kinematics, contact forces, and suspension system components.\n\n3. Critical Speed: It occurs at a traveling speed vh called the hunting speed, which depends on specific track and vehicle conditions.\n\n4. Consequences: Hunting leads to severe ride deterioration, increased wear, potential safety hazards, and in extreme cases, derailment risk.\n\n5. Mechanism: The conical geometry of railway wheels creates a self-centering mechanism at low speeds, but above the critical speed, this mechanism becomes unstable and produces growing lateral oscillations.\n\n6. Importance: Understanding and detecting hunting proximity is essential for maintaining safety, ride comfort, and extending rolling stock life cycle.",
            "context": "Hunting is a coupled lateral and yaw limit cycle oscillation of the vehicle, physically limited by wheel flanges. It occurs at the hunting speed vh depending on track/vehicle conditions. Arises from nonlinear dynamics associated with wheel-rail contact kinematics, contact forces, and suspension components. Critical for safety, ride comfort, and rolling stock life cycle.",
            "source": "elecricsemple.pdf"
        },
        {
            "question": "What is the Maximum Lyapunov Exponent?",
            "answer": "The Maximum Lyapunov Exponent (MLE) is a mathematical measure used in this study as follows:\n\n1. Purpose: It characterizes the rate at which nearby trajectories in a dynamical system diverge or converge over time.\n\n2. Interpretation:\n   - Negative MLE: Trajectories converge, indicating stable behavior.\n   - Zero MLE: Trajectories maintain constant separation, characteristic of a limit cycle (such as hunting).\n   - Positive MLE: Trajectories diverge, indicating chaotic or unstable behavior.\n\n3. Application to Hunting: As the vehicle speed approaches the hunting speed, the MLE evolves from negative values toward zero or slightly positive values, independently of vehicle and track conditions.\n\n4. Estimation Methods:\n   - Direct methods: Identify nearby state vectors and track their separation (sensitive to noise).\n   - Indirect methods: Estimate a nonlinear model, compute its Jacobian, then apply standard MLE computation (accuracy issues for high-order systems).\n   - Grey model approach (proposed): Uses a physics-informed model structure to overcome limitations of both direct and indirect methods.",
            "context": "The MLE characterizes whether two response trajectories converge (negative MLE), maintain constant separation as in a limit cycle (zero MLE), or diverge (positive MLE). As vehicle speed approaches hunting speed, MLE evolves from negative toward zero. Estimation via grey model approach overcomes limitations of direct and indirect data-driven methods.",
            "source": "elecricsemple.pdf"
        },
        # --- Conclusions ---
        {
            "question": "What conclusions does the paper draw?",
            "answer": "The paper draws the following conclusions:\n\n1. The feasibility of assessing proximity to hunting onset during normal vehicle operation has been examined for the first time, and the proposed MLE-based method demonstrates excellent performance.\n\n2. The grey-model-based MLE estimation approach successfully overcomes limitations of state-of-the-art data-driven methods when applied to high-order, complex systems like railway vehicles.\n\n3. The method achieves 97.08% classification accuracy across extensive Monte Carlo simulations, representing a significant improvement over alternative approaches.\n\n4. Robustness to faulty suspensions and hollow-worn wheels has been demonstrated, though some performance compromise exists under worn-track conditions.\n\n5. The capability to assess hunting proximity in real-time supports higher travel speeds while maintaining safe margins from hunting instability.\n\n6. Further experimental validation on operating railway networks remains necessary.",
            "context": "Conclusions: Feasibility of hunting proximity assessment demonstrated. 97.08% classification accuracy achieved. Grey model overcomes data-driven MLE limitations. Robustness to faulty conditions demonstrated. Performance compromise under worn-track conditions noted. Further experimental validation needed.",
            "source": "elecricsemple.pdf"
        },
        # --- Important Findings ---
        {
            "question": "What are the important findings?",
            "answer": "The important findings of this study are:\n\n1. 97.08% Accuracy: The proposed method achieves excellent classification accuracy in identifying hunting proximity intervals across all operating scenarios.\n\n2. First-of-its-Kind: This is the first study to examine the feasibility of assessing proximity to hunting onset during normal vehicle operation, rather than detecting hunting after it occurs.\n\n3. Grey Model Superiority: The physics-informed grey model approach significantly outperforms the alternative indirect data-driven MLE estimation method, which was also evaluated for comparison.\n\n4. MLE as Reliable Indicator: The Maximum Lyapunov Exponent consistently evolves from negative values toward zero as speed approaches the hunting threshold, providing a reliable proximity measure independent of operating conditions.\n\n5. Practical Robustness: The method maintains performance under realistic uncertainties including varying track irregularities, passenger loads, faulty suspensions, and hollow-worn wheels.\n\n6. Real-Time Potential: The approach is designed for on-board real-time implementation using standard acceleration sensors, making it practically deployable.",
            "context": "Key findings: 97.08% classification accuracy. First feasibility study of hunting proximity assessment. Grey model outperforms data-driven alternatives. MLE provides reliable condition-independent proximity measure. Robustness under realistic operating uncertainties demonstrated.",
            "source": "elecricsemple.pdf"
        },

        # --- Paraphrased/Alternative Questions for Railway Paper ---
        {
            "question": "Describe the simulation setup used in this study.",
            "answer": "The simulation setup includes:\n\n1. Vehicle Model: A 21-DOF multi-body model developed in MATLAB consisting of 7 rigid bodies (car-body, 2 bogies, 4 wheelsets).\n\n2. Track Model: Alignment track irregularity modeled as wide-sense stationary random signal following ERRI B176 standard.\n\n3. Contact Model: Wheel-rail contact parameters obtained from precomputed look-up tables using the equivalent-elastic method with FASTSIM creep force computation.\n\n4. Operating Scenarios: Four scenarios (A-D) with different suspension, wheel, and track conditions to test robustness.\n\n5. Monte Carlo Setup: For each scenario and proximity interval, simulations performed at randomly selected constant speeds with random track profiles.\n\n6. Sensors: 22 uni-axial accelerometers mounted at specified locations on the vehicle.\n\n7. Vehicle Parameters: Based on an Athens Metro vehicle.",
            "context": "21-DOF multi-body model in MATLAB. Track irregularity following ERRI B176 standard. Wheel-rail contact via equivalent-elastic method and FASTSIM. Four operating scenarios. Monte Carlo simulations at random speeds. 22 accelerometers. Athens Metro vehicle parameters.",
            "source": "elecricsemple.pdf"
        },
    ]

    # =================================================================
    # 2. TESTIMAGE.PDF — BLADE Multi-Agent Failure Attribution
    # =================================================================
    blade_qa = [
        {
            "question": "Tell me about this PDF.",
            "answer": "This PDF introduces BLADE (Blame Localization via Audited Dependency Evidence), a training-free framework developed by researchers at the University of Connecticut MINDS Lab for failure attribution in large language model (LLM) multi-agent systems.\n\nKey aspects of the paper:\n\n1. Problem: When multiple LLM agents collaborate on a task and fail, identifying which agent caused the failure is challenging because errors can propagate across agents.\n\n2. Innovation: BLADE moves beyond simply tracing where information came from to auditing how information was used at each decision point.\n\n3. Method: Uses probabilistic judgments via Jev to select and audit source-decision handoffs, then synthesizes a global diagnosis.\n\n4. Results: Evaluated on 1,228 failed trajectories across Who&When and Who&When Pro benchmarks, achieving the best or tied-best step accuracy without requiring reference answers.\n\n5. Authors: Xiaoguang Guo, Yaohao Long, and Chuxu Zhang.",
            "context": "BLADE: Auditing Information Use for Failure Attribution in LLM Multi-Agent Systems. Xiaoguang Guo, Yaohao Long, Chuxu Zhang. University of Connecticut. We introduce BLADE, a training-free framework that moves beyond tracing information to auditing its use. Evaluated on 1,228 failed trajectories.",
            "source": "testimage.pdf"
        },
        {
            "question": "Give me a summary of this PDF.",
            "answer": "This paper presents BLADE (Blame Localization via Audited Dependency Evidence), a framework for identifying the root cause of failures in multi-agent LLM systems. Summary:\n\n1. Motivation: In multi-agent systems, when the final output is wrong, determining which agent made the critical error is difficult because intermediate errors propagate through the chain.\n\n2. Key Insight: Knowing where information came from (tracing) is not the same as knowing where it went wrong (auditing). BLADE audits information use rather than just tracing dependencies.\n\n3. Three-Stage Process:\n   - Stage 1: Select plausible source-decision handoffs using Jev dependency probabilities.\n   - Stage 2: Audit whether receiving decisions faithfully use source information, distinguishing error introduction, propagation, and repair.\n   - Stage 3: Synthesize a global diagnosis across all execution steps.\n\n4. Evaluation: Tested on 1,228 failed runs on Who&When and Who&When Pro benchmarks, achieving best or tied-best step accuracy.\n\n5. Significance: Works without requiring reference answers or additional training.",
            "context": "BLADE audits information use for failure attribution in LLM multi-agent systems. Three stages: select handoffs via Jev, audit information use, synthesize global diagnosis. Evaluated on 1,228 failed trajectories on Who&When benchmarks.",
            "source": "testimage.pdf"
        },
        {
            "question": "What is the precise problem statement?",
            "answer": "The precise problem statement is failure attribution in multi-agent LLM systems:\n\n1. Setting: Multiple LLM agents collaborate on a complex task, each handling a subtask and passing results to downstream agents.\n\n2. Problem: When the final output is incorrect, the system needs to identify:\n   - Which specific agent is responsible for the failure.\n   - At which step the decisive error occurred.\n\n3. Challenge: Errors can be introduced at one step, propagated through subsequent steps, or even repaired by later agents. Simply tracing where information came from does not reveal where it went wrong.\n\n4. Requirements: The attribution method should work without reference answers (ground truth) and without additional model training.\n\n5. Goal: Provide accurate, automated failure diagnosis that can help debug and improve multi-agent system reliability.",
            "context": "Failure attribution seeks to identify the responsible agent and decisive error step in a multi-agent execution trajectory. Knowing where information came from is not the same as knowing where it went wrong. Must work without reference answers or additional training.",
            "source": "testimage.pdf"
        },
        {
            "question": "Explain the methodology.",
            "answer": "BLADE operates through three main stages:\n\n1. Handoff Selection (Stage 1):\n   - Identifies source-decision handoffs in the execution trace.\n   - Uses Jev (structured probabilistic judgments) to compute dependency probabilities.\n   - Selects the most relevant handoffs for detailed auditing.\n\n2. Information Use Auditing (Stage 2):\n   - For each selected handoff, examines whether the receiving decision used the source information faithfully.\n   - Classifies each handoff into categories: error introduction (new error created), error propagation (existing error passed through), or error repair (previous error corrected).\n   - This auditing step is the key innovation that distinguishes BLADE from simple tracing methods.\n\n3. Global Diagnosis Synthesis (Stage 3):\n   - Combines evidence from all audited handoffs.\n   - Produces a unified diagnosis identifying the responsible agent and the decisive error step.\n   - Considers the full execution context rather than individual handoffs in isolation.\n\nBLADE is training-free, meaning it does not require fine-tuning or additional model training to perform attribution.",
            "context": "BLADE operates in three stages: selecting plausible source-decision handoffs via Jev dependency probabilities, auditing whether receiving decisions use source information faithfully (distinguishing error introduction, propagation, and repair), and synthesizing a global diagnosis across all steps.",
            "source": "testimage.pdf"
        },
        {
            "question": "What are the limitations?",
            "answer": "The limitations of this study include:\n\n1. Benchmark Scope: Evaluation is limited to five subsets of Who&When and Who&When Pro benchmarks, not open-ended real-world multi-agent deployments.\n\n2. Recorded Traces: BLADE relies on recorded execution traces, which may not capture all relevant information about agent decision-making.\n\n3. Context Window: Input shortening to fit context windows may omit useful evidence that could improve attribution accuracy.\n\n4. Single Decision Model: Experiments use a single decision model (Jev), and performance may vary with different underlying models.\n\n5. Multi-Agent Topology: The evaluation focuses on specific multi-agent configurations and may not generalize to all possible agent interaction patterns.",
            "context": "Limitations: Evaluation on five subsets of Who&When benchmarks. Reliance on recorded execution traces. Input shortening may omit useful evidence. Experiments use a single decision model, Jev.",
            "source": "testimage.pdf"
        },
        {
            "question": "What are the advantages of BLADE?",
            "answer": "The advantages of BLADE include:\n\n1. Training-Free: Does not require any fine-tuning or additional model training, making it immediately deployable.\n\n2. Audit-Based Approach: Goes beyond simple information tracing to actually audit how information was used at each step, providing deeper failure understanding.\n\n3. High Accuracy: Achieves the best or tied-best step accuracy on the benchmark without requiring reference answers.\n\n4. Error Classification: Can distinguish between error introduction, propagation, and repair, providing nuanced diagnostic information.\n\n5. Global Diagnosis: Synthesizes evidence across all execution steps rather than making local judgments, improving attribution reliability.\n\n6. No Reference Answers: Operates without ground truth answers, making it applicable in real-world scenarios where correct answers are unknown.",
            "context": "BLADE is training-free, audits information use rather than just tracing, achieves best or tied-best step accuracy, distinguishes error types, synthesizes global diagnosis, and works without reference answers.",
            "source": "testimage.pdf"
        },
        {
            "question": "What are the important findings?",
            "answer": "The important findings from this paper are:\n\n1. Auditing vs Tracing: Simply tracing information flow is insufficient for failure attribution; auditing how information is used provides significantly better results.\n\n2. Superior Performance: BLADE achieves the best or tied-best step accuracy across all evaluated benchmarks without requiring reference answers.\n\n3. Error Taxonomy: The framework successfully distinguishes between error introduction, error propagation, and error repair, providing actionable diagnostic information.\n\n4. Practical Applicability: The training-free nature of BLADE makes it immediately usable for debugging multi-agent systems in production.\n\n5. Jev Effectiveness: Structured probabilistic judgments via Jev provide reliable dependency probability estimates for handoff selection.",
            "context": "Key findings: Auditing information use outperforms tracing. Best or tied-best step accuracy achieved. Successful error type classification. Training-free design enables practical deployment. Jev provides reliable dependency estimates.",
            "source": "testimage.pdf"
        },
    ]

    qa_list.extend(railway_qa)
    qa_list.extend(blade_qa)

    # =================================================================
    # 3. AUTO-GENERATED QA FROM PDF CHUNKS
    # =================================================================
    question_templates = [
        "What does the document discuss about {topic}?",
        "Explain {topic} as described in the document.",
        "What information does the paper provide about {topic}?",
        "Describe {topic} based on the document content.",
        "Summarize what the document says about {topic}.",
    ]

    for pdf_name in ["elecricsemple.pdf", "testimage.pdf"]:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found, skipping auto-generation")
            continue

        chunks = extract_pdf_chunks(pdf_path, max_words=150, overlap=30)
        for i, c in enumerate(chunks):
            words = c["text"].split()
            if len(words) < 30:
                continue

            # Extract meaningful topic phrases
            content_words = [w.strip(".,;:()[]{}") for w in words if len(w) > 3 and not w.isdigit()]
            stopwords = {"this", "that", "with", "from", "into", "have", "been", "were", "their",
                         "also", "which", "based", "using", "these", "those", "than", "more",
                         "other", "each", "between", "under", "over", "about", "such", "given",
                         "following", "through", "after", "before", "during"}
            content_words = [w for w in content_words if w.lower() not in stopwords]

            if len(content_words) < 3:
                continue

            # Pick topic from content words
            topic = " ".join(content_words[1:4]).lower()
            template_idx = i % len(question_templates)
            question = question_templates[template_idx].format(topic=topic)

            # Create answer from first ~50 words, cleaned up
            answer_words = words[:50]
            answer = " ".join(answer_words)
            if not answer.endswith("."):
                answer += "."

            qa_list.append({
                "question": question,
                "answer": answer,
                "context": c["text"],
                "source": pdf_name
            })

    # =================================================================
    # 4. NEGATIVE (OUT-OF-DOMAIN) EXAMPLES
    # =================================================================
    negative_questions = [
        "What is the capital city of France?",
        "Who is the president of Mars?",
        "What is the recipe for chocolate cake?",
        "Who won the FIFA World Cup in 2022?",
        "How do you change a car tire?",
        "What is the theory of relativity?",
        "Tell me about quantum computing.",
        "What are the symptoms of the flu?",
        "Explain machine learning algorithms.",
        "What is the population of Tokyo?",
    ]
    for nq in negative_questions:
        qa_list.append({
            "question": nq,
            "answer": "This information is not provided in the uploaded document. The document does not contain content related to this question. Please ask a question relevant to the document's content.",
            "context": "The uploaded document is a technical research paper. It does not cover general knowledge topics outside its specific research domain.",
            "source": "negative"
        })

    return qa_list


def prepare_and_save():
    print("=" * 65)
    print("PREPARING ENHANCED TRAINING DATASET (V2)")
    print("=" * 65)

    data = build_curated_qa_dataset()

    # Statistics
    curated = sum(1 for d in data if d["source"] != "negative" and "What does the document" not in d["question"]
                  and "Explain " not in d["question"][:10] and "What information" not in d["question"]
                  and "Describe " not in d["question"][:10] and "Summarize what" not in d["question"])
    auto = len(data) - curated - sum(1 for d in data if d["source"] == "negative")
    neg = sum(1 for d in data if d["source"] == "negative")

    print(f"Total training examples: {len(data)}")
    print(f"  Curated high-quality: {curated}")
    print(f"  Auto-generated: {auto}")
    print(f"  Negative (out-of-domain): {neg}")
    print(f"  Average answer length: {sum(len(d['answer'].split()) for d in data) / len(data):.1f} words")

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to: {OUTPUT_DATASET}")
    return data


if __name__ == "__main__":
    prepare_and_save()

