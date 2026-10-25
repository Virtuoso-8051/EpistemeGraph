# 🛡️ EpistemeGraph (`kg_audit`): Non-Euclidean Sheaf-Spectral & Calibrated MLOps Engine for LLM Hallucination Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Neo4j AuraDB Cloud](https://img.shields.io/badge/Neo4j-AuraDB_Cloud-008CC1.svg)](https://neo4j.com/cloud/aura/)
[![PyTorch](https://img.shields.io/badge/PyTorch-AttentionResNet-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-9_Model_Tournament-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Benchmark_UI-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-SQLite_Tracking-0194E2.svg)](https://mlflow.org/)

**Lead Researcher & Principal MLOps Architect:** Anurag Raj ([`@Virtuoso-8051`](https://github.com/Virtuoso-8051))  
**Core Stack:** Python 3.12 | Neo4j AuraDB Cloud | OpenAI (`gpt-5.4-mini` / `gpt-4o`) | PyTorch (`AttentionResNet` & `SentenceTransformers`) | Scikit-Learn (`StackingClassifier`, `CalibratedClassifierCV`) | SciPy & NetworkX | FastAPI & Uvicorn | Streamlit & PyVis | MLflow (`mlflow.db`)

---

## 1. Executive Summary & Research Motivation

Large Language Models (LLMs) routinely generate fluent, authoritative responses that fabricate technical mechanisms, conflate distinct mathematical formulations across papers, or violate relational constraints in source literature—a phenomenon known as **epistemic hallucination**. Existing Retrieval-Augmented Generation (RAG) and GraphRAG verification pipelines suffer from three foundational vulnerabilities:

1. **The LLM Overconfidence Paradox:** When queried on specialized scientific literature, an LLM routinely assigns a self-reported confidence score of **$91\%\text{–}99\%$** even when fabricating non-existent mathematical pipelines (e.g., asserting or hypothesizing Convolutional Neural Networks, Quantum Schrödinger circuits, or Causal Do-Calculus inside `node2vec` or `DeepWalk`), yielding an empirical discrimination **ROC-AUC of only $0.3419$** (worse than random guessing).
2. **The Cascading Ground-Truth Fallacy:** Standard Knowledge Graph (KG) auditors assume that the reference graph stored in the database is $100\%$ infallible. However, because an LLM constructs the initial Knowledge Graph from raw PDFs, unverified extractions, pronoun mentions, or incomplete chunk coverage in Phase 1 corrupt downstream verification in Phase 3.
3. **Euclidean Cosine Blindness:** Standard vector-similarity auditors rely on flat cosine similarity in $\mathbb{R}^d$, which cannot detect topological shortcut edges ("manifold holes"), directional relation inversions, or cross-document chimeric conflations where every individual word exists in the corpus but never co-occurs in the same technical context.

**EpistemeGraph (`kg_audit`)** resolves all three vulnerabilities through a **4-Phase Autonomous Verification & Self-Improving MLOps Architecture**:

* **Phase 1 — Full-Corpus Multi-PDF Knowledge Graph Construction & Unsupervised Fidelity Audit (`kg_builder.py`):** Parses $100\%$ of sentence-aligned chunks across uploaded scientific PDFs concurrently (5-thread parallel execution with SHA-256 disk caching), enforces cryptographic-level verbatim/anchored quote verification (rejecting $38.3\%$ of noisy LLM extractions before database ingestion), resolves cross-document synonyms and plural/parenthetical acronyms, pushes the canonical graph to **Neo4j AuraDB Cloud**, and computes a deterministic **4-Pillar Stage I Graph Fidelity Score ($S_{\text{KG}} = 79.43\%$)**.
* **Phase 2 — Single-Pass Polarity-Aware Query Interception (`llm_interceptor.py`):** Acts as an AI middleware gatekeeper that intercepts user queries, captures the LLM's natural language response alongside its **Self-Reported Accuracy Score ($P_{\text{LLM}}$)** and rationale, and extracts an ontology-normalized **Induced Response Graph ($G_{\text{resp}}$)** tagged with claim polarity (`ASSERTED`, `SPECULATIVE`, or `REFUTED`).
* **Phase 3 — CASSE Non-Euclidean Tensor, Lexical-Contextual Chunk Grounding & Bayesian Provenance Audit (`graph_retriever.py` & `truth_evaluator.py`):** Retrieves the focused $k$-hop ground-truth subgraph $G_{\text{GT}}$ from Neo4j Cloud, aligns entities onto a unified canonical vertex space $V_U$, combines **Curvature-Aware Sheaf Spectral Energy (CASSE)** with **IDF-Weighted Lexical-Contextual Chunk Co-Occurrence ($\Phi_{\text{cooc}}, \rho_{\text{OOC}}$)**, and extracts an **8-dimensional mathematical feature vector $\mathbf{X} \in \mathbb{R}^8$**.
* **Phase 4 — Autonomous 9-Model Deep Learning & MLOps Calibration Engine (`model_trainer.py`):** Evaluates $\mathbf{X}$ across a 9-architecture tournament—featuring a custom **PyTorch Feature-Level Self-Attention Residual Network (`PyTorch_AttentionResNet`)**, a **5-Model `StackingMetaEnsemble`**, and Platt-calibrated tree/kernel classifiers (`RandomForest`, `ExtraTrees`, `GradientBoosting`, `RBF_SVM`, `DeepMLP_NeuralNet`, `AdaBoost`, `LogisticRegression`). Using **L-BFGS-B Maximum-Likelihood Optimization**, it autonomously learns the 8 Boltzmann-Gibbs energy/prior multipliers and **Conformal Tri-State Decision Thresholds ($\tau_{\text{verify}}^*, \tau_{\text{abstain}}^*$)** without human intervention.

---

## 2. End-to-End System Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                     PHASE 1: FULL-CORPUS MULTI-PDF INGESTION & KG FIDELITY ENGINE                 |
|                                  (kg_audit/components/kg_builder.py)                              |
+---------------------------------------------------------------------------------------------------+
|  [data/uploads/paper1..3.pdf] ---> [Sentence-Aligned Chunker (76 Chunks)]                         |
|                                                  |                                                |
|                                    [5-Thread Parallel OpenAI Extractor]                           |
|                                    (SHA-256 Disk Cache: $0 Re-run Cost)                           |
|                                                  |                                                |
|                                                  v                                                |
|  [Verbatim/Anchored Quote Bouncer: Rejects 238/621 Ungrounded Claims]                             |
|                                                  |                                                |
|                                                  v                                                |
|  [Neo4j AuraDB Cloud (313 Nodes, 347 Unique Edges)] <--- [4-Pillar S_KG Audit (S_KG = 79.43%)]    |
+---------------------------------------------------------------------------------------------------+
                                                   |
                           Saves S_KG State & Full Corpus Chunk Index
                    (final_model/kg_fidelity_state.json & latest_chunks.json)
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------+
|                        USER QUERY INTERCEPTION (ui/streamlit_app.py / app.py)                     |
+---------------------------------------------------------------------------------------------------+
                       |                                                   |
         PATH 1: LLM Generation & Self-Audit                 PATH 2: Ground-Truth Subgraph Audit
     (kg_audit/components/llm_interceptor.py)              (kg_audit/components/graph_retriever.py)
                       |                                                   |
                       v                                                   v
       +-------------------------------+                   +-------------------------------+
       | - Raw Natural Language Answer |                   | - Exact/Acronym Seed Match    |
       | - Self-Reported Score (P_LLM) |                   | - 2-Hop Geodesic Subgraph G   |
       | - Polarity-Tagged Graph G'    |                   | - Sparse Adjacency Matrix     |
       | - SHA-256 Query Cache ($0)    |                   | - Provenance & Quote Metadata |
       +-------------------------------+                   +-------------------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                     PHASE 3: CASSE NON-EUCLIDEAN & PROVENANCE EVALUATOR                           |
|                              (kg_audit/components/truth_evaluator.py)                             |
+---------------------------------------------------------------------------------------------------+
|  1. Canonical Entity Alignment (V_U) + IDF Chunk Co-Occurrence (Φ_cooc) & OOC Rate (ρ_OOC)        |
|  2. 8D Invariant Vector X = [E_sheaf, ΔK, D_spec, T_src, F_time, I_indep, C_conf, P_LLM]^T        |
|  3. Fuses Boltzmann-Gibbs Score + model.pkl Probability + Corpus Support ---> S_Query             |
|  4. Compound Trust Score: S_Final = (S_Query ^ η*) * (S_KG_eff ^ (1 - η*))                        |
|  5. Conformal Tri-State Verdict: [ VERIFIED (>=0.60) | ABSTAIN (0.28-0.60) | HALLUCINATED (<0.28)]|
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|                   PHASE 4: AUTONOMOUS 9-MODEL DEEP CALIBRATION & MLOPS LOOP                       |
|                               (kg_audit/components/model_trainer.py)                              |
+---------------------------------------------------------------------------------------------------+
|  - Logs Live 8D Telemetry Vectors ---> [data/live_telemetry_claims.csv]                           |
|  - Trains 9 Architectures (PyTorch_AttentionResNet, StackingMetaEnsemble, RF, ExtraTrees, GBDT,   |
|    RBF_SVM, DeepMLP_NeuralNet, AdaBoost, LogisticRegression) with Platt Sigmoid Calibration       |
|  - L-BFGS-B Optimizer learns Boltzmann Multipliers (w_E*, w_P*) & Conformal Thresholds (τ_v*, τ_a*)|
|  - Promotes Winning Champion to [final_model/model.pkl & production_model_metrics.json]           |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Complete Mathematical Formulation

### 3.1 Phase 1: Unsupervised 4-Pillar Knowledge Graph Fidelity ($S_{\text{KG}}$)

Before trusting a Knowledge Graph $G_{\text{GT}} = (V_{\text{GT}}, E_{\text{GT}})$ to audit an LLM, `kg_builder.py` evaluates its structural and semantic fidelity against the raw source chunks $\mathcal{C} = \{c_1, c_2, \dots, c_m\}$ without requiring human ground-truth annotations:

1. **Provenance Grounding Precision ($\Phi_{\text{prov}}$):** For each raw extracted claim $e_i = (u_i, r_i, v_i)$ with supporting quote $q_i$ from chunk $c_i$, let $s_{\text{quote}}(q_i, c_i) \in [0, 1]$ denote the normalized longest-common-substring / sentence-window anchor similarity, $\text{ent}(e_i, q_i, c_i) \in [0, 1]$ denote the stem-level entity presence in the quote, and $\text{sem}(e_i, q_i) = \text{clip}\left(\frac{\cos(\mathbf{e}_{\text{triple}}, \mathbf{e}_{q_i}) - 0.15}{0.50}, 0, 1\right)$ denote the dense all-MiniLM-L6-v2 semantic alignment. Claims failing verification ($s_{\text{quote}} < 0.82$ and anchor $< 0.62$, or $\text{ent} < 0.85$) are rejected into `final_model/rejected_claims.json`. Over all $N_{\text{raw}}$ extracted claims:
   $$\Phi_{\text{prov}} = \frac{1}{N_{\text{raw}}} \sum_{i=1}^{N_{\text{raw}}} \nu_i \cdot \left(0.40 \cdot \text{ent}(e_i, q_i, c_i) + 0.60 \cdot \text{sem}(e_i, q_i)\right)$$
   *where $\nu_i = 1.0$ for verbatim matches and $\nu_i = 0.88 \cdot s_{\text{anchor}}$ for paraphrase-anchored matches.*

2. **Information-Theoretic Chunk Coverage Recall ($\Phi_{\text{cov}}$):** Measures how thoroughly the verified triplets $E(c_j)$ capture the sentence-level semantic information of each chunk $c_j$ across all $m$ document chunks (including zero-claim chunks), weighted by logarithmic chunk capacity $w_j = \ln(1 + \vert{}c_j\vert{})$:
   $$\Phi_{\text{cov}} = \frac{\sum_{j=1}^{m} w_j \cdot \left(\frac{1}{\vert{}S(c_j)\vert{}} \sum_{s \in S(c_j)} \min\left(1, \frac{\max_{t \in E(c_j)} \cos(\mathbf{e}_s, \mathbf{e}_t)}{\tau_{\text{cov}}}\right)\right)}{\sum_{j=1}^{m} w_j}, \quad \tau_{\text{cov}} = 0.60$$

3. **Entity-Resolved Topological Coherence ($\Phi_{\text{topo}}$):** After bidirectional acronym expansion (e.g., matching "LINE" $\leftrightarrow$ "Large-scale Information Network Embedding"), plural normalization, and cosine entity resolution ($\ge 0.85$), let $V_{\text{LCC}}$ be the Largest Connected Component and $V_{\text{clust}}$ be vertices in components of size $\ge 3$:
   $$\Phi_{\text{topo}} = \left(0.5 + 0.25 \frac{\vert{}V_{\text{LCC}}\vert{}}{\vert{}V\vert{}} + 0.25 \frac{\vert{}V_{\text{clust}}\vert{}}{\vert{}V\vert{}}\right) \cdot \left(1 - \frac{\vert{}E_{\text{self-loop}}\vert{}}{\vert{}E_{\text{verified}}\vert{}}\right)$$

4. **Intra-Document Functional Consistency ($\Phi_{\text{cons}}$):** Penalizes multi-target contradictions across functional predicates ($E_{\text{conflict}}$):
   $$\Phi_{\text{cons}} = 1 - \frac{\vert{}E_{\text{conflict}}\vert{}}{\vert{}E_{\text{verified}}\vert{}}$$

The unified Stage I KG Fidelity Score ($S_{\text{KG}} \in [0, 1]$) fuses Precision and Recall via their harmonic mean, regularized by topology and consistency:
$$S_{\text{KG}} = \left( \frac{2 \cdot \Phi_{\text{prov}} \cdot \Phi_{\text{cov}}}{\Phi_{\text{prov}} + \Phi_{\text{cov}} + \epsilon} \right)^{0.70} \cdot \left( \Phi_{\text{topo}} \cdot \Phi_{\text{cons}} \right)^{0.30}$$

### 3.2 Phase 3: Lexical-Contextual Chunk Grounding & The 8D CASSE Feature Vector ($\mathbf{X}$)

Given a user query $Q$, the LLM's induced response graph $G_{\text{resp}} = (V_{\text{resp}}, E_{\text{resp}})$, and the retrieved Neo4j ground-truth subgraph $G_{\text{GT}} = (V_{\text{GT}}, E_{\text{GT}})$, `truth_evaluator.py` first computes two domain-agnostic corpus grounding signals from `final_model/latest_chunks.json`:

* **Out-of-Corpus Technical Stem Ratio ($\rho_{\text{OOC}} \in [0, 1]$):** Let $U(Q) = \{s_1, \dots, s_k\}$ be the unique ASCII-folded, stemmed technical tokens in $Q$, and let $\mathcal{S}_{\text{corpus}} = \bigcup_{j=1}^{m} \mathcal{S}(c_j)$ be the global vocabulary of technical stems across all $m$ PDF chunks:
  $$\rho_{\text{OOC}}(Q) = \frac{\left\vert{}\{s \in U(Q) : s \notin \mathcal{S}_{\text{corpus}}\}\right\vert{}}{\vert{}U(Q)\vert{}}$$

* **IDF-Weighted Chunk Co-Occurrence & Bigram Continuity ($\Phi_{\text{cooc}} \in [0, 1]$):** Assigning each query stem an Inverse Document Frequency specificity weight $w(s) = \left[\log_2(2 + \text{df}(s))\right]^{-1}$ and letting $\mathcal{B}(Q)$ be adjacent technical stem bigrams in $Q$:
  $$\Phi_{\text{cooc}}(Q) = 0.75 \cdot \max_{c_j \in \mathcal{C}} \left( \frac{\sum_{s \in U(Q) \cap \mathcal{S}(c_j)} w(s)}{\sum_{s \in U(Q)} w(s)} \right) + 0.25 \cdot \frac{\left|\mathcal{B}(Q) \cap \mathcal{B}_{\text{corpus}}\right|}{\vert{}\mathcal{B}(Q)|}$$

Next, `truth_evaluator.py` aligns $G_{\text{GT}}$ and $G_{\text{resp}}$ onto a canonical vertex space $V_U$, constructs the induced ground-truth reference graph $\widetilde{G}_{\text{GT}}$ over active response entities $V_{\text{act}}$, and extracts the 8-dimensional invariant vector $\mathbf{X}$:
$$\mathbf{X} = \begin{bmatrix} E_{\text{sheaf}}, & \Delta K, & D_{\text{spec}}, & T_{\text{source}}, & F(\Delta t), & I_{\text{indep}}, & C_{\text{conflict}}, & P_{\text{LLM}} \end{bmatrix}^T$$

1. **Cellular Sheaf Coboundary Dirichlet Energy ($x_1 = E_{\text{sheaf}}$):** Assigns unit-normalized 384D semantic embeddings $\mathbf{x}_u \in \mathbb{S}^{383}$ to vertex stalks $\mathcal{F}(u)$. For each asserted edge $e = (u, r_{\text{resp}}, v) \in E_{\text{resp}}$, let $\tilde{\mathbf{u}}_e = \frac{\mathbf{x}_u + 0.5 \mathbf{r}_{\text{resp}}}{\|\mathbf{x}_u + 0.5 \mathbf{r}_{\text{resp}}\|_2}$. For supported edges $(u, v) \in E(\widetilde{G}_{\text{GT}})$, the restriction map $\mathcal{F}_{u \to e}$ is the exact Householder Orthogonal Reflection Matrix $\mathbf{H}_e = \mathbf{I} - 2 \frac{\mathbf{d}_e \mathbf{d}_e^T}{\|\mathbf{d}_e\|_2^2}$ where $\mathbf{d}_e = \tilde{\mathbf{u}}_e^{\text{GT}} - \mathbf{x}_v$; for unsupported edges, $\mathcal{F}_{v \to e}(\mathbf{x}_v) = \mu_u \mu_v \Phi_{\text{cooc}}(1 - \rho_{\text{OOC}})\mathbf{x}_v$:
   $$E_{\text{sheaf}} = \frac{1}{2 |E_{\text{resp}}|} \sum_{e=(u,v) \in E_{\text{resp}}} \left\| \mathcal{F}_{u \to e}(\tilde{\mathbf{u}}_e) - \mathcal{F}_{v \to e}(\mathbf{x}_v) \right\|_2^2 + \frac{1}{2}\left(0.55(1 - \Phi_{\text{cooc}}) + 0.45 \rho_{\text{OOC}}\right)$$

2. **Bounded Forman-Ricci Curvature Distortion ($x_2 = \Delta K$):** Computes discrete Forman-Ricci edge curvature on $\widetilde{G}_{\text{GT}}$ and $G_{\text{resp}}$:
   $$\text{Ric}_G(e) = w_e \left( \frac{w_u}{w_e} + \frac{w_v}{w_e} - \sum_{e_u \sim e} \frac{w_u}{\sqrt{w_e w_{e_u}}} - \sum_{e_v \sim e} \frac{w_v}{\sqrt{w_e w_{e_v}}} \right)$$
   Verified edges incur bounded hyperbolic tangent distortion $0.18 \tanh\left(\frac{\vert{}\text{Ric}_{\text{GT}}(e) - \text{Ric}_{\text{resp}}(e)\vert{}}{6}\right)$, whereas hallucinated shortcut edges incur manifold tearing penalty $0.55 + 0.40 \sigma(-\text{Ric}_{\text{resp}}(e)) + 0.30 \rho_{\text{OOC}}$, regularized by corpus support.

3. **Wasserstein-Frobenius Spectral Laplacian Divergence ($x_3 = D_{\text{spec}}$):** Over the Symmetric Normalized Laplacians $\mathbf{L}_{\text{norm}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ of $\widetilde{G}_{\text{GT}}$ and $G_{\text{resp}}$:
   $$D_{\text{spec}} = 0.40 \cdot W_1\left(\boldsymbol{\lambda}(\widetilde{G}_{\text{GT}}), \boldsymbol{\lambda}(G_{\text{resp}})\right) + 0.25 \cdot \frac{\left\Vert{} \mathbf{L}_{\text{norm}}(\widetilde{G}_{\text{GT}}) - \mathbf{L}_{\text{norm}}(G_{\text{resp}}) \right\Vert{}_F}{\vert{}V_{\text{act}}\vert{}}$$

4. **Conjugate Beta-Bernoulli Bayesian Source Trust ($x_4 = T_{\text{source}}$):** Posterior expectation under a $\text{Beta}(1, 1)$ prior combining chunk-level grounding and subgraph edge verification rate $(1 - r_{\text{unv}})$:
   $$T_{\text{source}} = \frac{1 + 12 \Phi_{\text{cooc}}(1 - \rho_{\text{OOC}}) + 4(1 - r_{\text{unv}})}{2 + 12 \Phi_{\text{cooc}}(1 - \rho_{\text{OOC}}) + 4(1 - r_{\text{unv}}) + 12 C_{\text{conflict}} + 6 \rho_{\text{OOC}}}$$

5. **Weibull-Exponential Temporal & Speculative Volatility Decay ($x_5 = F(\Delta t)$):**
   $$F(\Delta t) = \exp\left(-\left(0.03 + 0.22 \rho_{\text{spec}} + 0.25 C_{\text{conflict}}\right) \cdot \left(0.40 + 0.80 C_{\text{conflict}}\right)\right)$$

6. **Multi-Document Source Independence ($x_6 = I_{\text{indep}}$):**
   $$I_{\text{indep}} = \text{clip}\left(0.20 + 0.75 \Phi_{\text{cooc}}(1 - 0.80 \rho_{\text{OOC}}), \; 0.12, \; 0.96\right)$$

7. **Active Topological-Lexical Conflict Ratio ($x_7 = C_{\text{conflict}}$):** Couples structural edge conflict $r_{\text{unv}} = \frac{\vert{}E_{\text{resp}} \setminus E(\widetilde{G}_{\text{GT}})\vert{}}{\vert{}E_{\text{resp}}\vert{}}$ with corpus premise conflict:
   $$C_{\text{conflict}} = 0.40 \cdot r_{\text{unv}} + 0.60 \cdot \text{clip}\left(0.65(1 - \Phi_{\text{cooc}}) + 0.75 \rho_{\text{OOC}}, \; 0, \; 1\right)$$

8. **LLM Self-Reported Confidence Prior ($x_8 = P_{\text{LLM}}$):** Raw self-estimated probability returned by the language model during Phase 2 interception.

### 3.3 Phase 4: Self-Learned Multipliers, Hybrid Belief Inference & Conformal Thresholds

In Phase 4 (`model_trainer.py`), a 9-architecture tournament trains on the 8D feature vectors $\mathbf{X}$, while an L-BFGS-B Maximum-Likelihood Optimizer initialized by the PyTorch_AttentionResNet self-attention gate $\boldsymbol{\alpha}(\mathbf{X}) = \text{Softmax}(\mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{X}))$ learns the optimal energy penalty weights $\mathbf{w}_{\mathcal{H}}^* = [0.0800, 0.0800, 0.0500, 0.1500]^T$ and normalized reliability prior weights $\mathbf{w}_{\mathcal{P}}^* = [0.8721, 0.0581, 0.0581, 0.0116]^T$:

1. **Boltzmann-Gibbs Analytical Score ($S_{\text{Boltz}}$):**
   $$S_{\text{Boltz}}(\mathbf{X}) = \left(w_{p,1}^* T_{\text{source}} + w_{p,2}^* F(\Delta t) + w_{p,3}^* I_{\text{indep}} + w_{p,4}^* P_{\text{LLM}}(1 - C_{\text{conflict}})\right) \cdot \exp\left(-\left(w_{e,1}^* E_{\text{sheaf}} + w_{e,2}^* \Delta K + w_{e,3}^* D_{\text{spec}} + w_{e,4}^* C_{\text{conflict}}\right)\right)$$
   *(Note: The optimizer autonomously assigns $w_{p,4}^* = 0.0116$ ($1.16\%$) to $P_{\text{LLM}}$ and $w_{p,1}^* = 0.8721$ ($87.21\%$) to $T_{\text{source}}$, mathematically proving that the system learns to distrust LLM self-confidence!)*

2. **Hybrid Query Subgraph Belief ($S_{\text{Query}}$) & Compound Trust Score ($S_{\text{Final}}$):** Let $P_{\text{ML}}(Y=1 \mid \mathbf{X})$ be the Platt-calibrated probability from `final_model/model.pkl` and $S_{\text{corpus}} = \text{clip}(1.18 \Phi_{\text{cooc}} - 0.62 \rho_{\text{OOC}} + 0.18, 0.05, 0.96)$:
   $$S_{\text{Query}} = 0.25 \cdot S_{\text{Boltz}}(\mathbf{X}) + 0.20 \cdot P_{\text{ML}}(Y=1 \mid \mathbf{X}) + 0.55 \cdot S_{\text{corpus}}$$
   $$S_{\text{Final}} = \left(S_{\text{Query}}\right)^{\eta^*} \cdot \left(\min\left(1.0, \frac{S_{\text{KG}}}{0.75}\right)\right)^{1 - \eta^*}, \quad \eta^* = 0.85$$

3. **Learned Conformal Tri-State Decision Rule ($\tau_{\text{verify}}^* = 0.6000, \; \tau_{\text{abstain}}^* = 0.2768$):**
   $$\text{Verdict}(Q) = \begin{cases} \text{\bf VERIFIED} & \text{if } S_{\text{Final}} \ge \tau_{\text{verify}}^* \text{ and } \Phi_{\text{cooc}} \ge 0.435 \text{ and } \rho_{\text{OOC}} \le 0.160 \\ \text{\bf ABSTAIN} & \text{if } S_{\text{Final}} \ge \tau_{\text{abstain}}^* \text{ and not } (\rho_{\text{OOC}} \ge 0.220 \wedge \Phi_{\text{cooc}} \le 0.210) \\ \text{\bf HALLUCINATED} & \text{otherwise} \end{cases}$$

---

## 4. Complete Repository Directory Structure

For full scientific reproducibility, this repository includes the pre-computed `Artifacts/`, `final_model/`, and `data/` benchmark records (while keeping `.env` private via `.gitignore`):

```plaintext
EpistemeGraph/
├── .vscode/                              # VS Code workspace settings
├── Artifacts/                            # Timestamped MLOps execution artifacts & zero-cost caches
│   ├── <MM_DD_YYYY_HH_MM_SS>/            # Run-specific directories (e.g., 10_25_2026_10_40_21)
│   │   ├── kg_builder/                   # Exported extracted_knowledge_graph.json (313 nodes, 375 edges)
│   │   ├── llm_interceptor/              # Captured query, LLM response, polarity triples & telemetry JSON
│   │   ├── graph_retriever/              # Focused 2-hop Neo4j subgraph extracted_subgraph.json
│   │   ├── truth_evaluator/              # Edge-by-edge CASSE audit accuracy_report.json
│   │   └── model_trainer/                # Trained challenger model.pkl & calibration_metrics.json
│   ├── llm_cache/                        # SHA-256 Phase 1 chunk extraction cache ($0 API re-runs)
│   ├── query_cache/                      # SHA-256 Phase 2 query interception cache ($0 benchmark re-runs)
│   └── pyvis_kg.html                     # Interactive physics-based Knowledge Graph visualization
├── assets/                               # Architecture diagrams & UI screenshots
├── data/                                 # Research corpus, 100-query benchmark & live MLOps telemetry
│   ├── uploads/                          # Input PDFs: paper1.pdf (LINE), paper2.pdf (node2vec), paper3.pdf (DeepWalk)
│   ├── benchmark_100_queries.txt         # 100 structured benchmark queries (40 Factual, 20 Borderline, 40 Hallucination)
│   ├── benchmark_100_results.csv         # Complete 100-query empirical evaluation table (97% 3-class, 99% binary accuracy)
│   └── live_telemetry_claims.csv         # Auto-logged 8D query feature vectors for continuous self-retraining
├── final_model/                          # Production model & persistent Knowledge Graph state registry
│   ├── model.pkl                         # Promoted Champion Platt-Calibrated Classifier (RandomForest, AUC=0.9932)
│   ├── kg_fidelity_state.json            # Persistent Phase 1 S_KG (79.43%) & 29 cross-PDF shared entities
│   ├── latest_chunks.json                # Full 76-chunk sentence-aligned corpus index across paper1..3.pdf
│   ├── production_model_metrics.json     # 9-Model Tournament leaderboard & learned multipliers/thresholds
│   └── rejected_claims.json              # Audit log of 238 ungrounded LLM claims rejected during Phase 1
├── kg_audit/                             # Core installable Python research package
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── kg_builder.py                 # Phase 1: Full-PDF chunker, 5-thread OpenAI extractor, quote bouncer & Neo4j push
│   │   ├── llm_interceptor.py            # Phase 2: Polarity-aware query interceptor & SHA-256 query cache
│   │   ├── graph_retriever.py            # Phase 3A: Seed-entity & 2-hop geodesic Neo4j AuraDB subgraph retriever
│   │   ├── truth_evaluator.py            # Phase 3B: CASSE tensor, IDF chunk grounding & Conformal tri-state evaluator
│   │   └── model_trainer.py              # Phase 4: 9-Architecture Tournament (PyTorch AttentionResNet + 8 Sklearn models)
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py              # Pipeline & component configuration dataclasses
│   │   └── artifact_entity.py            # Strictly-typed stage output dataclasses
│   ├── exceptions/
│   │   ├── __init__.py
│   │   └── exception.py                  # Custom KGAuditException with filename & line number tracing
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger.py                     # Timestamped execution logging
│   ├── pipelines/
│   │   └── evaluation_pipeline.py        # End-to-end Phase 1-4 orchestrator with MLflow SQLite tracking
│   └── utils/
│       ├── __init__.py
│       └── main_utils.py                 # Serialization & file I/O helpers
├── logs/                                 # Runtime .log files
├── mlruns/                               # MLflow experiment tracking store
├── notebooks/                            # Interactive Jupyter mathematical verification notebooks
├── ui/
│   └── streamlit_app.py                  # Dual-Path Streamlit UI + Live 100-Query Auto-Runner
├── .env                                  # Confidential API credentials (excluded from Git via .gitignore)
├── .gitignore                            # Git exclusion rules
├── app.py                                # FastAPI REST Middleware Server (/build-kg and /verify endpoints)
├── benchmark_100_queries.py              # CLI batch runner for 100-query evaluation & ablation summary
├── mlflow.db                             # SQLite relational backend for MLflow tracking
├── requirements.txt                      # Python package dependencies
└── setup.py                              # Local editable package installer (kg_audit)
```

---

## 5. Empirical Research Results (3-Paper Corpus & 100-Query Benchmark)

We evaluated EpistemeGraph across three foundational network embedding papers uploaded simultaneously into `data/uploads/`:
* **Document 1 (`paper1.pdf`)**: LINE: Large-scale Information Network Embedding (Tang et al., WWW 2015).
* **Document 2 (`paper2.pdf`)**: node2vec: Scalable Feature Learning for Networks (Grover & Leskovec, KDD 2016).
* **Document 3 (`paper3.pdf`)**: DeepWalk: Online Learning of Social Representations (Perozzi et al., KDD 2014).

### 5.1 Phase 1: Full-Corpus Multi-PDF Knowledge Graph Construction Telemetry

Across 76 sentence-aligned chunks ($100\%$ of document text excluding bibliographies; $82,170$ prompt tokens and $102,575$ completion tokens), KGBuilder extracted 621 raw claims, rejected 238 ungrounded claims ($236$ `entities_not_in_quote`, $2$ `entities_not_in_anchor`), resolved 313 canonical entities, and pushed 375 verified claims ($347$ unique merged edges) to Neo4j AuraDB Cloud:

| Phase 1 Fidelity Metric | Symbol | Empirical Value | Technical Interpretation |
| :--- | :---: | :---: | :--- |
| Overall Stage I KG Fidelity | $S_{\text{KG}}$ | **79.43%** | High-integrity ground-truth baseline across all 76 chunks. |
| Provenance Grounding Precision | $\Phi_{\text{prov}}$ | **86.92%** | Strict quote verification ($38.3\%$ noisy LLM claims filtered out). |
| Chunk Coverage Recall | $\Phi_{\text{cov}}$ | **69.81%** | Sentence-level semantic retention across all 76 PDF chunks (0 empty chunks). |
| Topological Coherence | $\Phi_{\text{topo}}$ | **84.74%** | Strong Giant-Component connectivity across 313 resolved entities. |
| Functional Consistency | $\Phi_{\text{cons}}$ | **99.47%** | Near-zero functional contradictions across 375 verified claims. |

**29 Cross-PDF Shared Entities Automatically Linked Across Papers:** Large-scale Information Network Embedding, DEEPWALK, SKIPGRAM, random walk, truncated random walk, depth-first search, hierarchical softmax, classical stochastic gradient descent, asynchronous stochastic gradient algorithm, objective function, Optimization Problem, Node Classification, Link Prediction, multi-label classification, one-vs-rest logistic regression classifier, ISOMAP, matrix factorization, embedding, low-dimensional space, vertex representation, representation learning, unsupervised feature learning, mapping function, representation, CLUSTERS, social network, social relationship, FLICKR, and BlogCatalog.

### 5.2 Phase 2 & 3: 100-Query Benchmark Results (`data/benchmark_100_results.csv`)

We evaluated all 100 queries in `data/benchmark_100_queries.txt` (40 FACTUAL single/cross-paper queries, 20 BORDERLINE unstated/chimeric/out-of-scope queries, and 40 HALLUCINATION adversarial traps):

**Confusion Matrix (97.0% Three-Class Accuracy | 99.0% Binary Hallucination Detection Accuracy)**

| Ground-Truth Category (Expected Verdict) | Predicted VERIFIED | Predicted ABSTAIN | Predicted HALLUCINATED | Category Recall |
| :--- | :---: | :---: | :---: | :---: |
| **FACTUAL** (Q001–Q040 $\rightarrow$ Expected: VERIFIED) | 40 | 0 | 0 | 40 / 40 (100.0%) |
| **BORDERLINE** (Q041–Q060 $\rightarrow$ Expected: ABSTAIN) | 2 (Q049, Q058) | 17 | 1 (Q048) | 17 / 20 (85.0%) |
| **HALLUCINATION** (Q061–Q100 $\rightarrow$ Expected: HALLUCINATED) | 0 | 0 | 40 | 40 / 40 (100.0%) |
| **Total (100 Queries)** | **42** | **17** | **41** | **97 / 100 (97.0%)** |

**Category-Wise Distribution of All 8 Invariants & Compound Scores (Mean $\pm$ SD)**

| Metric / Invariant | Symbol | FACTUAL (n=40) | BORDERLINE (n=20) | HALLUCINATION (n=40) | 100-Query ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ChatGPT Self-Reported Accuracy | $P_{\text{LLM}}$ ($x_8$) | $0.9178 \pm 0.0522$ | $0.8520 \pm 0.1059$ | $0.9258 \pm 0.0774$ | **0.3419** (Fails) |
| Cellular Sheaf Energy (↓) | $E_{\text{sheaf}}$ ($x_1$) | $0.2906 \pm 0.1248$ | $0.5404 \pm 0.1272$ | $0.7451 \pm 0.0735$ | 0.9773 |
| Forman-Ricci Distortion (↓) | $\Delta K$ ($x_2$) | $0.2994 \pm 0.1091$ | $0.5580 \pm 0.1524$ | $0.7887 \pm 0.0918$ | 0.9671 |
| Spectral Laplacian Div. (↓) | $D_{\text{spec}}$ ($x_3$) | $0.1766 \pm 0.0696$ | $0.2614 \pm 0.0996$ | $0.3316 \pm 0.0690$ | 0.8746 |
| Bayesian Source Trust (↑) | $T_{\text{source}}$ ($x_4$) | $0.7289 \pm 0.0987$ | $0.4101 \pm 0.1252$ | $0.1630 \pm 0.0526$ | 0.9971 |
| Temporal Freshness (↑) | $F(\Delta t)$ ($x_5$) | $0.9399 \pm 0.0265$ | $0.8541 \pm 0.0485$ | $0.7427 \pm 0.0446$ | 0.9858 |
| Source Independence (↑) | $I_{\text{indep}}$ ($x_6$) | $0.6933 \pm 0.0938$ | $0.4202 \pm 0.0786$ | $0.2526 \pm 0.0208$ | 0.9992 |
| Active Conflict Ratio (↓) | $C_{\text{conflict}}$ ($x_7$) | $0.2628 \pm 0.1178$ | $0.5674 \pm 0.1417$ | $0.8563 \pm 0.0882$ | 0.9933 |
| Query Subgraph Belief (↑) | $S_{\text{Query}}$ | $0.8762 \pm 0.0645$ | $0.5818 \pm 0.1035$ | $0.1904 \pm 0.0924$ | 0.9996 |
| EpistemeGraph Final Score (↑) | $S_{\text{Final}}$ | $0.8935 \pm 0.0562$ | $0.6299 \pm 0.0945$ | $0.2407 \pm 0.0999$ | **0.9996** |
| Calibration Gap ($S_{\text{Final}} - P_{\text{LLM}}$) | $\Delta_{\text{cal}}$ | **$-2.43\%$** | **$-22.21\%$** | **$-68.50\%$** | Brier: 0.0619 vs 0.3554 |

### 5.3 Phase 4: Autonomous 9-Model Tournament Leaderboard (`final_model/production_model_metrics.json`)

| Rank | Algorithm / Architecture | ROC-AUC (↑) | F1-Score (↑) | Brier Loss (↓) | ECE (↓) | Composite Score (↑) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 1 | **RandomForest (Promoted Champion)** | **0.9932** | 0.9902 | **0.0111** | **0.0057** | **0.9765** |
| 🥈 2 | PyTorch_AttentionResNet (Self-Attention DL) | 0.9913 | 0.9902 | 0.0117 | 0.0061 | 0.9734 |
| 🥉 3 | ExtraTrees | 0.9895 | 0.9902 | 0.0110 | 0.0058 | 0.9727 |
| 4 | StackingMetaEnsemble (5-Model Meta-Learner) | 0.9885 | 0.9902 | 0.0111 | 0.0063 | 0.9710 |
| 5 | GradientBoosting | 0.9891 | 0.9902 | 0.0129 | 0.0131 | 0.9630 |
| 6 | RBF_SVM | 0.9885 | 0.9878 | 0.0125 | 0.0130 | 0.9630 |
| 7 | AdaBoost | 0.9821 | 0.9902 | 0.0155 | 0.0177 | 0.9489 |
| 8 | DeepMLP_NeuralNet (128-64-32) | 0.9842 | 0.9580 | 0.0348 | 0.0350 | 0.9144 |
| 9 | LogisticRegression | 0.9851 | 0.9752 | 0.0254 | 0.0518 | 0.9079 |

---

## 6. Installation, Confidential `.env` Setup & Execution Guide

**Step 1: Clone the Repository & Create a Python 3.12 Virtual Environment**
```bash
git clone [https://github.com/Virtuoso-8051/EpistemeGraph.git](https://github.com/Virtuoso-8051/EpistemeGraph.git)
cd EpistemeGraph
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 2: Configure Your Private `.env` File**
Because `.env` contains secret API credentials, it is excluded from Git via `.gitignore`. Create a `.env` file in the root directory `EpistemeGraph/.env`:
```env
OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY"
OPENAI_MODEL="gpt-5.4-mini"

NEO4J_URI="neo4j+s://<your-instance-id>.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="<your-neo4j-auradb-password>"
NEO4J_DATABASE="neo4j"
```

**Step 3: Launch the Interactive Streamlit Dashboard & Live 100-Query Auto-Runner**
```bash
streamlit run ui/streamlit_app.py --server.fileWatcherType none
```
* **Zero-Cost Instant Reproduction:** Because `Artifacts/llm_cache/`, `Artifacts/query_cache/`, and `final_model/` are included in the repository, you can re-run the entire 100-query benchmark in the UI ("▶️ Auto-Run Benchmark Live on UI") at **$0.00** OpenAI API cost!
* **Command-Line Auto-Run Mode:**
  ```bash
  streamlit run ui/streamlit_app.py --server.fileWatcherType none -- --autorun
  ```
* **Testing on New Custom PDFs:** Click "🧹 Clear Staged PDFs in `data/uploads/`" in the left sidebar, upload any new PDF(s), click "🚀 Construct Full KG & Overwrite Neo4j Cloud", and type any research query in the Phase 2 & 3 arena.

**Step 4: Launch the FastAPI REST Middleware Server (Optional)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
* Swagger UI documentation is available at `http://localhost:8000/docs`.

---

## 7. Citation

```bibtex
@article{raj2026epistemegraph,
  author       = {Anurag Raj},
  title        = {EpistemeGraph: A Two-Stage Cellular Sheaf, Forman-Ricci Curvature, and Conformal MLOps Framework for Hallucination Detection over Cloud Knowledge Graphs},
  journal      = {arXiv preprint},
  year         = {2026},
  url          = {[https://github.com/Virtuoso-8051/EpistemeGraph](https://github.com/Virtuoso-8051/EpistemeGraph)}
}
```