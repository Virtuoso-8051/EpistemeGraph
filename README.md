# 🛡️ EpistemeGraph (`kg_audit`): Non-Euclidean Knowledge Graph Hallucination & Trust Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Neo4j AuraDB Cloud](https://img.shields.io/badge/Neo4j-AuraDB_Cloud-008CC1.svg)](https://neo4j.com/cloud/aura/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dual_View_UI-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-SQLite_Tracking-0194E2.svg)](https://mlflow.org/)

<!-- **Lead Researcher & Principal MLOps Architect:** Anurag Raj ([`@Virtuoso-8051`](https://github.com/Virtuoso-8051))   -->
**Core Stack:** Python 3.12 | Neo4j AuraDB Cloud | OpenAI (`gpt-5.4-mini` / `gpt-4o`) | FastAPI & Uvicorn | Streamlit & PyVis | Scikit-Learn & PyTorch (`SentenceTransformers`) | MLflow (`mlflow.db`)

---

## 1. Executive Summary & Research Motivation

Large Language Models (LLMs) frequently generate fluent, high-confidence assertions that fabricate facts or violate relational constraints present in source literature—a phenomenon known as **hallucination**. Existing Retrieval-Augmented Generation (RAG) and GraphRAG verification pipelines suffer from two foundational vulnerabilities:

1. **The LLM Overconfidence Paradox:** When queried on domain-specific scientific literature, an LLM will routinely report a self-estimated confidence score of **96%–98%** even while fabricating non-existent architectural mechanisms.
2. **The Cascading Ground-Truth Fallacy:** Standard Knowledge Graph (KG) auditors assume that the graph stored in the database is $100\%$ infallible. However, because an LLM constructs the initial Knowledge Graph from raw documents, extraction errors or incomplete coverage in Phase 1 corrupt downstream verification in Phase 3.

**EpistemeGraph (`kg_audit`)** addresses both vulnerabilities through a **4-Phase Autonomous Verification & Self-Improving MLOps Architecture**:

* **Phase 1 — Multi-PDF Knowledge Graph Construction & Unsupervised Fidelity Audit (`kg_builder.py`):** Parses and chunks multiple scientific PDFs concurrently (5-thread parallel execution with SHA-256 disk caching), extracts ontology-normalized claims with verbatim quote verification, resolves cross-document entities and acronyms, pushes the unified graph to **Neo4j AuraDB Cloud**, and computes a deterministic **Phase 1 Graph Fidelity Score ($S_{\text{KG}}$)**.
* **Phase 2 — Single-Pass Query Interception & Self-Assessment (`llm_interceptor.py`):** Acts as an AI middleware gatekeeper that intercepts user queries, captures the LLM's natural language response alongside its **Self-Reported Accuracy Score ($P_{\text{LLM}}$)** and confidence rationale, and extracts an induced response graph $\mathcal{G}'$.
* **Phase 3 — CASSE Non-Euclidean Tensor & Provenance Audit (`graph_retriever.py` & `truth_evaluator.py`):** Retrieves the active ground-truth subgraph $\mathcal{G}$ from Neo4j Cloud and evaluates structural and semantic alignment using **Curvature-Aware Sheaf Spectral Energy (CASSE)** combined with Bayesian provenance metrics to construct an 8-dimensional feature vector $\mathbf{X}$.
* **Phase 4 — Autonomous Self-Improving MLOps Calibration Engine (`model_trainer.py`):** Passes $\mathbf{X}$ into a Platt-calibrated Machine Learning / Deep Neural Network classifier (`final_model/model.pkl`) selected across 7 candidate architectures via `GridSearchCV`. Live query telemetry is logged to `data/live_telemetry_claims.csv` to continuously retrain and promote challenger models via **MLflow (`mlflow.db`)**.

---

## 2. End-to-End System Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                        PHASE 1: MULTI-PDF INGESTION & KG FIDELITY ENGINE                          |
|                                  (kg_audit/components/kg_builder.py)                              |
+---------------------------------------------------------------------------------------------------+
|  [data/uploads/*.pdf] ---> [High-Density Chunker] ---> [5-Thread Parallel OpenAI Extractor]       |
|                                                                  |                                |
|                                                    (SHA-256 Disk Cache: $0 Re-run)                |
|                                                                  v                                |
|  [Neo4j AuraDB Cloud] <--- [4-Pillar S_KG Audit] <--- [Acronym & Embedding Entity Resolution]     |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                    Saves S_KG & Graph State
                                  (final_model/kg_fidelity_state.json)
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                           USER QUERY INTERCEPTION (ui/streamlit_app.py / app.py)                  |
+---------------------------------------------------------------------------------------------------+
                       |                                                   |
         PATH 1: LLM Generation & Self-Audit                 PATH 2: Ground-Truth Subgraph Audit
     (kg_audit/components/llm_interceptor.py)              (kg_audit/components/graph_retriever.py)
                       |                                                   |
                       v                                                   v
       +-------------------------------+                   +-------------------------------+
       | - Raw Natural Language Answer |                   | - Keyword/Alias Cypher Match  |
       | - Self-Reported Score (P_LLM) |                   | - Active Subgraph G (Neo4j)   |
       | - Induced Response Graph G'   |                   | - Sparse Adjacency Matrix     |
       +-------------------------------+                   +-------------------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                     PHASE 3: CASSE NON-EUCLIDEAN & PROVENANCE EVALUATOR                           |
|                              (kg_audit/components/truth_evaluator.py)                             |
+---------------------------------------------------------------------------------------------------+
|  Computes 8D Feature Vector X = [E_sheaf, ΔK, D_spec, T_src, F_time, I_indep, C_conf, P_LLM]      |
|  Evaluates Calibrated Belief S_Query via final_model/model.pkl                                    |
|  Computes Final Compound Trust: S_Final = (S_Query ^ 0.80) * (S_KG ^ 0.20)                        |
|  Outputs Tri-State Verdict: [ VERIFIED (>=75%) | ABSTAIN (50%-75%) | HALLUCINATED (<50%) ]        |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                   PHASE 4: AUTONOMOUS SELF-IMPROVING MLOPS CALIBRATION LOOP                       |
|                               (kg_audit/components/model_trainer.py)                              |
+---------------------------------------------------------------------------------------------------+
|  Appends X + Topological Oracle Label ---> [data/live_telemetry_claims.csv]                       |
|  Trains 7 ML/DL Models (LogReg, RF, ExtraTrees, GBDT, AdaBoost, DeepMLP_NeuralNet, RBF_SVM)       |
|  Champion vs. Challenger Gatekeeper: Promotes to final_model/model.pkl if Composite Score Wins    |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Formulation

### 3.1 Phase 1: Unsupervised 4-Pillar Knowledge Graph Fidelity ($S_{\text{KG}}$)

Before using a Knowledge Graph to audit an LLM, `kg_builder.py` quantifies the structural and semantic fidelity of the extracted graph $\mathcal{G} = (V, E)$ against the raw source chunks $\{c_1, c_2, \dots, c_m\}$ without requiring manual ground-truth labels:

1. **Provenance Grounding Precision ($\Phi_{\text{prov}}$):** Evaluates whether the verbatim supporting quote $q_i$ for each extracted triplet $e_i = (u_i, r_i, v_i)$ physically exists in chunk $c_i$ ($\mathbb{I}_{\text{quote}} \in [0, 1]$) weighted by the dense cosine similarity between the triplet embedding and quote embedding:
   $$\Phi_{\text{prov}} = \frac{1}{\vert{}E_{\text{raw}}\vert{}} \sum_{i=1}^{\vert{}E_{\text{raw}}\vert{}} \mathbb{I}_{\text{quote}}(q_i, c_i) \cdot \max\left(0, \cos\left(\mathbf{e}_{(u_i, r_i, v_i)}, \mathbf{e}_{q_i}\right)\right)$$

2. **Information-Theoretic Chunk Coverage Recall ($\Phi_{\text{cov}}$):** Measures how thoroughly the extracted triplets capture the semantic content of each chunk $c_j$, weighted by logarithmic chunk length $w_j = \log(1 + \vert{}c_j\vert{})$:
   $$\Phi_{\text{cov}} = \frac{\sum_{j=1}^{m} w_j \cdot \max\left(0, \cos\left(\mathbf{e}_{c_j}, \mathbf{e}_{\mathcal{G}(c_j)}\right)\right)}{\sum_{j=1}^{m} w_j}$$

3. **Entity-Resolved Topological Coherence ($\Phi_{\text{topo}}$):** After merging cross-document synonyms and acronyms (e.g., matching initials of `"LINE"` to `"Large-scale Information Network Embedding"` and `"BFS"` to `"breadth-first search"`), measures the fraction of entities residing in the Largest Connected Component ($V_{\text{LCC}}$) penalized by self-loops:
   $$\Phi_{\text{topo}} = \left(0.5 + 0.5 \frac{\vert{}V_{\text{LCC}}\vert{}}{\vert{}V\vert{}}\right) \cdot \left(1 - \frac{\vert{}E_{\text{self-loop}}\vert{}}{\vert{}E\vert{}}\right)$$

4. **Intra-Document Functional Consistency ($\Phi_{\text{cons}}$):** Penalizes unresolved contradictions across single-valued functional relations ($E_{\text{conflict}}$):
   $$\Phi_{\text{cons}} = 1 - \frac{\vert{}E_{\text{conflict}}\vert{}}{\vert{}E\vert{}}$$

The unified **Phase 1 KG Fidelity Score ($S_{\text{KG}} \in [0, 1]$)** combines Precision and Recall via their harmonic mean, regularized by topology and consistency:

$$S_{\text{KG}} = \left( \frac{2 \cdot \Phi_{\text{prov}} \cdot \Phi_{\text{cov}}}{\Phi_{\text{prov}} + \Phi_{\text{cov}} + \epsilon} \right)^{0.7} \cdot \left( \Phi_{\text{topo}} \cdot \Phi_{\text{cons}} \right)^{0.3}$$

---

### 3.2 Phase 3: Curvature-Aware Sheaf Spectral Energy (CASSE) & 8D Feature Vector ($\mathbf{X}$)

Given the canonical entity alignment between the active Neo4j ground-truth subgraph $\mathcal{G}_{\text{GT}}$ and the LLM's induced response graph $\mathcal{G}_{\text{resp}}$, `truth_evaluator.py` extracts an 8-dimensional feature vector $\mathbf{X}$:

$$\mathbf{X} = \begin{bmatrix} \mathcal{E}_{\text{sheaf}}, & \Delta \mathbf{K}, & D_{\text{spec}}, & T_{\text{source}}, & F(\Delta t), & I_{\text{indep}}, & C_{\text{conflict}}, & P_{\text{LLM}} \end{bmatrix}^T$$

1. **Cellular Sheaf Coboundary Residual ($\mathcal{E}_{\text{sheaf}}$):** Assigns 384-dimensional dense semantic embeddings (`all-MiniLM-L6-v2`) to vertex stalks $\mathcal{F}(u)$ and evaluates restriction map residuals over response edges $e = (u, v) \in E_{\text{resp}}$:
   $$\mathcal{E}_{\text{sheaf}}(\mathcal{G}_{\text{resp}} \mid \mathcal{G}_{\text{GT}}) = \frac{1}{\vert{}E_{\text{resp}}\vert{}} \sum_{(u,v) \in E_{\text{resp}}} \Vert{}\mathbf{\mathcal{F}}_{u \to e}(x_u) - \mathbf{\mathcal{F}}_{v \to e}(x_v)\Vert{}_2^2$$
2. **Bounded Forman-Ricci Curvature Distortion ($\Delta \mathbf{K}$):** Computes discrete edge curvature on the active subgraph to detect hallucinated shortcut edges ("topological holes"):
   $$\text{Ric}(e) = w_e \left( \frac{w_u}{w_e} + \frac{w_v}{w_e} - \sum_{e_u \sim e} \frac{w_u}{\sqrt{w_e w_{e_u}}} - \sum_{e_v \sim e} \frac{w_v}{\sqrt{w_e w_{e_v}}} \right)$$
   Unverified shortcut edges receive a bounded sigmoid manifold distortion penalty $\sigma(-\text{Ric}(e))$.
3. **Induced Spectral Laplacian Divergence ($D_{\text{spec}}$):** Computes the 1D Wasserstein Earth Mover's Distance ($W_1$) and normalized Frobenius norm between the eigenvalue spectra of the Symmetric Normalized Graph Laplacians $\mathbf{L}_{\text{norm}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ over the active response entity set:
   $$D_{\text{spec}} = 0.4 \cdot W_1(\boldsymbol{\lambda}_{\text{GT}}, \boldsymbol{\lambda}_{\text{resp}}) + 0.25 \cdot \frac{\Vert{}\mathbf{L}_{\text{norm}}(\mathcal{G}_{\text{GT}}) - \mathbf{L}_{\text{norm}}(\mathcal{G}_{\text{resp}})\Vert{}_F}{\vert{}V_{\text{active}}\vert{}}$$
4. **Bayesian Source Trust ($T_{\text{source}}$):** Posterior mean under a $\text{Beta}(1, 1)$ prior over historical source verification outcomes:
   $$E[\text{Beta}(\alpha, \beta)] = \frac{1 + S_{\text{correct}}}{2 + S_{\text{correct}} + S_{\text{incorrect}}}$$
5. **Temporal Freshness Decay ($F(\Delta t)$):** Exponential time decay $F(\Delta t) = e^{-\lambda \Delta t}$ parameterized by document age $\Delta t$ and relation volatility $\lambda$.
6. **Source Independence ($I_{\text{indep}}$):** Effective independent document cluster ratio.
7. **Active Conflict Ratio ($C_{\text{conflict}}$):** Fraction of asserted response edges that are topologically disconnected and semantically ungrounded ($<\tau_{\text{sim}}$) relative to $\mathcal{G}_{\text{GT}}$.
8. **LLM Self-Reported Confidence ($P_{\text{LLM}}$):** The raw self-estimated accuracy probability returned by the language model during Phase 2 interception.

---

### 3.3 Calibrated Belief Inference & Compound Trust Fusion ($S_{\text{Final}}$)

The 8D feature vector $\mathbf{X}$ is evaluated by the production Platt-calibrated classifier (`final_model/model.pkl`) to yield the query-level belief probability $S_{\text{Query}} = P(Y = 1 \mid \mathbf{X}) \in [0, 1]$. This is fused with the Phase 1 Graph Fidelity score $S_{\text{KG}}$ via **Confidence-Weighted Geometric Fusion**:

$$S_{\text{Final}} = \left(S_{\text{Query}}\right)^{0.80} \cdot \left(\max(0.10, S_{\text{KG}})\right)^{0.20}$$

---

## 4. Complete Repository Directory Layout

```text
EpistemeGraph/
├── .vscode/                              # VS Code workspace configurations
├── Artifacts/                            # Timestamped MLOps execution artifacts
│   ├── <MM_DD_YYYY_HH_MM_SS>/            # Run-specific directories (e.g., 10_18_2026_19_57_56)
│   │   ├── kg_builder/                   # Exported extracted_knowledge_graph.json
│   │   ├── llm_interceptor/              # Captured query, LLM response & triples JSON
│   │   ├── graph_retriever/              # Extracted Neo4j subgraph JSON
│   │   ├── truth_evaluator/              # CASSE accuracy_report.json
│   │   └── model_trainer/                # Trained model & calibration_metrics.json
│   ├── llm_cache/                        # SHA-256 chunk extraction cache ($0 API re-runs)
│   └── pyvis_kg.html                     # Compiled interactive physics-based KG view
├── assets/                               # Static project documentation & visual assets
├── data/                                 # Local data & live MLOps telemetry store
│   ├── uploads/                          # Multi-PDF input directory (paper1.pdf, paper2.pdf, paper3.pdf)
│   └── live_telemetry_claims.csv         # Auto-logged 8D query vectors for continuous retraining
├── final_model/                          # Production model & persistent KG state registry
│   ├── model.pkl                         # Promoted Champion Calibrated Classifier
│   ├── kg_fidelity_state.json            # Persistent Phase 1 S_KG & cross-PDF telemetry
│   └── production_model_metrics.json     # Champion composite score & 7-model leaderboard
├── kg_audit/                             # Core installable Python research package
│   ├── __init__.py                       # Package initializer
│   ├── components/                       # Modular pipeline execution engines
│   │   ├── __init__.py                   # Components initializer
│   │   ├── kg_builder.py                 # Phase 1: Parallel PDF chunker, OpenAI extractor, S_KG & Neo4j batch push
│   │   ├── llm_interceptor.py            # Phase 2: Query interceptor, self-reported confidence & G' extractor
│   │   ├── graph_retriever.py            # Phase 3A: Keyword/alias Neo4j AuraDB subgraph retriever
│   │   ├── truth_evaluator.py            # Phase 3B: CASSE tensor, 8D feature generator & compound scorer
│   │   └── model_trainer.py              # Phase 4: 7-model GridSearchCV + Platt calibration & auto-promoter
│   ├── entity/                           # Strictly-typed data contracts
│   │   ├── __init__.py                   # Entity initializer
│   │   ├── config_entity.py              # Environment & path configurations
│   │   └── artifact_entity.py            # Dataclass outputs passed between stages
│   ├── exceptions/                       # Centralized error traceback handling
│   │   ├── __init__.py                   # Exceptions initializer
│   │   └── exception.py                  # Custom KGAuditException (script name + line number)
│   ├── logging/                          # Observability & audit logs
│   │   ├── __init__.py                   # Logging initializer
│   │   └── logger.py                     # Timestamped file logger
│   ├── pipelines/                        # End-to-end MLOps orchestration
│   │   └── evaluation_pipeline.py        # Connects Phases 1-4 with MLflow SQLite tracking
│   └── utils/                            # Helper utilities
│       ├── __init__.py                   # Utils initializer
│       └── main_utils.py                 # Shared I/O and serialization helpers
├── logs/                                 # Runtime execution .log files
├── mlruns/                               # MLflow experiment run artifacts
├── notebooks/                            # Interactive Jupyter research & mathematical dry-runs
│   └── test_casse.ipynb                  # Isolated CASSE tensor & ModelTrainer verification notebook
├── ui/                                   # Interactive Dual-View Web Frontend
│   └── streamlit_app.py                  # Streamlit dashboard with PyVis graph & side-by-side audit arena
├── .env                                  # Secret credentials (OpenAI API & Neo4j AuraDB Cloud)
├── .gitignore                            # Git exclusion rules
├── app.py                                # FastAPI REST server (/build-kg and /verify endpoints)
├── mlflow.db                             # SQLite relational backend for MLflow tracking
├── requirements.txt                      # Python package dependencies
└── setup.py                              # Editable local package installer (kg_audit v0.0.1)
```

---

## 5. Empirical Research Results & Live Case Studies

We evaluated **EpistemeGraph** on two foundational network embedding papers uploaded simultaneously into `data/uploads/`:
* **Document 1 (`paper1.pdf`):** *LINE: Large-scale Information Network Embedding* (Tang et al., 2015).
* **Document 2 (`paper2.pdf`):** *node2vec: Scalable Feature Learning for Networks* (Grover & Leskovec, 2016).

### 5.1 Phase 1 Multi-PDF Knowledge Graph Construction Telemetry
Across **20 high-density chunks** (`10` per paper), the parallel `KGBuilder` engine extracted **160 verified triplets**, resolved **40 canonical entities**, and pushed **41 nodes** (`40` `:Entity` nodes + `1` `:GraphMeta` node) and **132 unique merged relationships** to **Neo4j AuraDB Cloud** (Instance ID: `e80fcda6`):

| Phase 1 Fidelity Metric | Mathematical Symbol | Empirical Value | Interpretation |
| :--- | :---: | :---: | :--- |
| **Overall KG Fidelity** | $S_{\text{KG}}$ | **50.24%** | Realistic zero-shot baseline across dense mathematical papers. |
| **Provenance Grounding Precision** | $\Phi_{\text{prov}}$ | **65.30%** | Verbatim quote verification + triplet-to-sentence cosine similarity. |
| **Chunk Coverage Recall** | $\Phi_{\text{cov}}$ | **44.00%** | Information-theoretic retention across sampled document chunks. |
| **Topological Coherence** | $\Phi_{\text{topo}}$ | **45.10%** | Giant-component connectivity across resolved entities. |
| **Functional Consistency** | $\Phi_{\text{cons}}$ | **100.00%** | Zero internal functional contradictions across extracted claims. |

**Cross-PDF Shared Entities Automatically Linked (5):** `Large-scale Information Network Embedding`, `embedding very large information networks into low-dimensional vector spaces`, `the limitation of the classical stochastic gradient descent`, `breadth-first search strategy`, and `Depth-First Search`.

---

### 5.2 Phase 2 & 3 Dual-Path Interception Benchmarks (Factual vs. Hallucination Trap)

| Evaluation Dimension | Case Study A: Cross-Paper Factual Query | Case Study B: Adversarial Hallucination Trap (CNNs & Self-Attention in node2vec) |
| :--- | :--- | :--- |
| **User Query** | *"How do LINE and node2vec approach information network embedding, and how do breadth-first search and depth-first search relate to their sampling strategies?"* | *"Explain how node2vec uses Convolutional Neural Networks and multi-head self-attention mechanisms to optimize its embedding vectors."* |
| **Path 1: ChatGPT Self-Reported Accuracy ($P_{\text{LLM}}$)** | **97.00%** | **98.00%** *(Severely Overconfident)* |
| **Neo4j Cloud Subgraph Nodes Retrieved** | **37 Nodes** | **35 Nodes** |
| **Sheaf Coboundary Energy ($\mathcal{E}_{\text{sheaf}}$)** | **0.2678** *(Low = Semantically Consistent)* | **0.6466** *(2.41x Spike = Unverified Extrapolation)* |
| **Forman-Ricci Distortion ($\Delta \mathbf{K}$)** | **0.4956** | **0.8237** *(Severe Manifold Deformation)* |
| **Active Conflict / Unverified Ratio ($C_{\text{conflict}}$)** | **20.0%** | **62.5%** *(5/8 Response Triples Ungrounded)* |
| **Query Subgraph Belief ($S_{\text{Query}}$)** | **57.65%** *(Pre-Active Subgraph Projection)* $\rightarrow$ **>85% (Post-Projection)** | **9.58%** |
| **Final Compound Trust ($S_{\text{Final}}$)** | **56.08% (`ABSTAIN`)** $\rightarrow$ **>78% (`VERIFIED`)** | **13.34% (`HALLUCINATION DETECTED`)** |
| **Calibration Gap ($S_{\text{Final}} - P_{\text{LLM}}$)** | $-40.92\%$ | **$-84.66\%$ vs. LLM Claim** |

---

## 6. Installation & Local Execution Guide

### Step 1: Clone the Repository & Create a Python 3.12 Virtual Environment
*(Note: Use Python 3.11 or 3.12 so pre-built C++/Fortran wheels for `scipy`, `numpy`, and `pandas` install cleanly).*

```bash
git clone [https://github.com/Virtuoso-8051/EpistemeGraph.git](https://github.com/Virtuoso-8051/EpistemeGraph.git)
cd EpistemeGraph
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Root `.env` Credentials
Create a `.env` file in the root directory with your OpenAI API key and Neo4j AuraDB Free Cloud credentials:

```env
OPENAI_API_KEY="sk-proj-YOUR_OPENAI_KEY"
OPENAI_MODEL="gpt-5.4-mini"

NEO4J_URI="neo4j+s://<your-instance-id>.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="<your-neo4j-cloud-password>"
NEO4J_DATABASE="neo4j"
```

### Step 3: Launch the Interactive Dual-View Streamlit Dashboard
Run Streamlit with the file watcher disabled so PyTorch/Transformers modules load cleanly without background inspection warnings:

```bash
streamlit run ui/streamlit_app.py --server.fileWatcherType none
```
* Open **`http://localhost:8501`** in your browser.
* Upload your PDFs in the left sidebar and click **"🚀 Construct KG & Push to Neo4j Cloud"**.
* Enter any research question in the **Phase 2 & 3 Query Arena** and click **"🔍 Intercept Query & Run CASSE Verification"**.

### Step 4: Launch the FastAPI REST Middleware Server (Optional)
To expose the programmatic REST endpoints (`POST /build-kg` and `POST /verify`) for external applications:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger UI documentation is available at **`http://localhost:8000/docs`**.

---

## 7. Author & Citation

Developed by **Anurag Raj** ([`@Virtuoso-8051`](https://github.com/Virtuoso-8051)). If referencing this architecture or the **Curvature-Aware Sheaf Spectral Energy (CASSE)** verification framework in academic work, please cite:

```bibtex
@software{epistemegraph,
  author       = {Anurag},
  title        = {EpistemeGraph: A Non-Euclidean Sheaf-Spectral and Calibrated MLOps Architecture for LLM Hallucination Detection over Cloud Knowledge Graphs},
  year         = {2026},
  url          = {[https://github.com/Virtuoso-8051/EpistemeGraph](https://github.com/Virtuoso-8051/EpistemeGraph)}
}
```