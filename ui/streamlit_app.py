import os
import sys
import json
import shutil
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline

st.set_page_config(
    page_title="EpistemeGraph | AI Hallucination & KG Audit Engine",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        color: #f9fafb;
        margin-bottom: 12px;
    }
    .verdict-verified {
        background-color: #065f46;
        border-left: 6px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        color: #ecfdf5;
        font-weight: 600;
    }
    .verdict-abstain {
        background-color: #78350f;
        border-left: 6px solid #f59e0b;
        padding: 15px;
        border-radius: 8px;
        color: #fffbeb;
        font-weight: 600;
    }
    .verdict-hallucinated {
        background-color: #7f1d1d;
        border-left: 6px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        color: #fef2f2;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ EpistemeGraph: Non-Euclidean KG Hallucination & Trust Engine")
st.caption(
    "Dual-Path Middleware: Phase 1 Full-PDF Knowledge Graph Fidelity ($S_{KG}$) "
    "× Phase 3 CASSE Topological & Calibrated ML Verification ($S_{Query}$)"
)

uploads_dir = os.path.join("data", "uploads")
os.makedirs(uploads_dir, exist_ok=True)
fidelity_state_path = os.path.join("final_model", "kg_fidelity_state.json")

# =====================================================================
# SIDEBAR: PHASE 1 MULTI-PDF INGESTION & NEO4J CLOUD PUSH
# =====================================================================
with st.sidebar:
    st.header("📂 Phase 1: Document Ingestion")
    st.write("Upload your research PDFs (`paper1.pdf`, `paper2.pdf`, etc.) to build & audit the Ground-Truth Knowledge Graph across 100% of document chunks.")

    if st.button("🧹 Clear Staged PDFs in `data/uploads/`", use_container_width=True):
        for fn in os.listdir(uploads_dir):
            fp = os.path.join(uploads_dir, fn)
            if os.path.isfile(fp):
                os.remove(fp)
        st.success("Cleared `data/uploads/` directory!")

    uploaded_files = st.file_uploader(
        "Select PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    saved_file_paths = []
    if uploaded_files:
        for uf in uploaded_files:
            dest_path = os.path.join(uploads_dir, uf.name)
            with open(dest_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_file_paths.append(dest_path)
        st.success(f"Staged {len(saved_file_paths)} file(s) in `data/uploads/`")

    existing_files = [
        os.path.join(uploads_dir, fn)
        for fn in sorted(os.listdir(uploads_dir))
        if fn.lower().endswith((".pdf", ".txt"))
    ]
    st.markdown(f"**Files currently in `data/uploads/` ({len(existing_files)}):**")
    for ef in existing_files:
        st.code(os.path.basename(ef), language="text")

    build_btn = st.button("🚀 Construct Full KG & Overwrite Neo4j Cloud", use_container_width=True, type="primary")

    if build_btn:
        target_files = saved_file_paths if saved_file_paths else existing_files
        if not target_files:
            st.error("Please upload at least one PDF or place files in `data/uploads/`!")
        else:
            with st.spinner("Chunking 100% of PDFs, extracting triplets via OpenAI, resolving acronyms/entities, and overwriting Neo4j AuraDB Cloud..."):
                pipeline = EvaluationPipeline()
                kg_artifact = pipeline.run_phase1_kg_construction(file_paths=target_files)
                st.session_state["latest_kg_artifact"] = kg_artifact
                st.session_state["latest_kg_json_path"] = kg_artifact.extracted_kg_file_path
            st.success("Phase 1 Complete! Full KG stored locally & pushed to Neo4j Cloud.")

# =====================================================================
# SECTION 1: PHASE 1 KNOWLEDGE GRAPH FIDELITY & TOPOLOGY VIEWER
# =====================================================================
if os.path.exists(fidelity_state_path):
    with open(fidelity_state_path, "r", encoding="utf-8") as f:
        kg_state = json.load(f)

    with st.expander("📊 Phase 1: Ground-Truth Knowledge Graph Fidelity ($S_{KG}$) & Cross-PDF Telemetry", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Overall KG Fidelity (S_KG)", f"{kg_state['overall_kg_accuracy_score'] * 100:.2f}%")
        c2.metric("Provenance Precision (Φ_prov)", f"{kg_state['provenance_grounding_score'] * 100:.1f}%")
        c3.metric("Chunk Coverage Recall (Φ_cov)", f"{kg_state['semantic_coverage_score'] * 100:.1f}%")
        c4.metric("Topological Coherence (Φ_topo)", f"{kg_state['topological_coherence_score'] * 100:.1f}%")
        c5.metric("Functional Consistency (Φ_cons)", f"{kg_state['internal_consistency_score'] * 100:.1f}%")

        m1, m2, m3, m4 = st.columns(4)
        m1.info(f"**Chunks Processed:** {kg_state['total_chunks_processed']}")
        m2.info(f"**Verified Triplets in Neo4j:** {kg_state['verified_triplets_pushed']} / {kg_state['raw_triplets_extracted']}")
        m3.info(f"**Canonical Entities:** {kg_state['resolved_entity_count']}")
        m4.info(f"**Cross-PDF Shared Entities:** {kg_state.get('cross_pdf_shared_entities_count', 0)}")

        shared_list = kg_state.get("cross_pdf_shared_entities", [])
        if shared_list:
            st.markdown(f"**🔗 Entities Linked Across Multiple PDFs:** `{', '.join(shared_list[:25])}`")

        html_path = os.path.join("Artifacts", "pyvis_kg.html")
        latest_kg_path = st.session_state.get("latest_kg_json_path") or kg_state.get("latest_extracted_kg_path")

        if latest_kg_path and os.path.exists(latest_kg_path):
            with open(latest_kg_path, "r", encoding="utf-8") as f:
                kg_data = json.load(f)

            net = Network(height="420px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
            net.barnes_hut(gravity=-3500, central_gravity=0.3, spring_length=140)

            for node in kg_data.get("nodes", []):
                pdfs = node.get("source_pdfs", [])
                is_shared = len(pdfs) > 1
                color = "#f59e0b" if is_shared else "#38bdf8"
                title_html = f"<b>{node['name']}</b><br>Type: {node.get('type')}<br>PDFs: {', '.join(pdfs)}<br>Aliases: {', '.join(node.get('aliases', []))}"
                net.add_node(node["name"], label=node["name"], title=title_html, color=color, size=24 if is_shared else 16)

            for edge in kg_data.get("edges", []):
                u, v = edge.get("subject"), edge.get("object")
                rel = edge.get("relation", "USES")
                quote = edge.get("evidence_quote", "")
                if u and v:
                    net.add_edge(u, v, label=rel, title=f"[{edge.get('source_name')}] {quote}", color="#64748b")

            os.makedirs("Artifacts", exist_ok=True)
            net.save_graph(html_path)

        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=440)
            st.caption("🟠 **Orange Nodes** = Shared concepts found across multiple PDFs | 🔵 **Blue Nodes** = Single-PDF entities (Hover over edges to read verbatim PDF evidence quotes)")

st.divider()

# =====================================================================
# SECTION 2: DUAL-VIEW QUERY INTERCEPTION & CASSE AUDIT ARENA
# =====================================================================
st.subheader("⚡ Phase 2 & 3: Live Query Interception & Dual-Path Mathematical Audit")

user_query = st.text_area(
    "Ask any research question across your uploaded PDFs:",
    placeholder="e.g., How do LINE and node2vec approach information network embedding, and how do breadth-first search and depth-first search relate to their sampling strategies?",
    height=90
)

verify_btn = st.button("🔍 Intercept Query & Run CASSE Verification", type="primary")

if verify_btn and user_query.strip():
    with st.spinner("Path 1: Querying OpenAI & capturing self-reported confidence | Path 2: Pulling Focused Neo4j Subgraph & computing CASSE Tensor..."):
        pipeline = EvaluationPipeline()
        interceptor_art, retriever_art, eval_art = pipeline.run_pipeline(user_query=user_query.strip())

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("### 🤖 Path 1: ChatGPT Raw Response & Self-Assessment")
        st.info(interceptor_art.llm_response)

        llm_acc_pct = interceptor_art.llm_self_reported_accuracy * 100.0
        st.metric(
            label="ChatGPT Self-Reported Accuracy Score",
            value=f"{llm_acc_pct:.2f}%",
            help="The confidence score ChatGPT assigned to its own response."
        )
        st.caption(f"**ChatGPT's Confidence Rationale:** *{interceptor_art.llm_confidence_reasoning}*")

        with open(interceptor_art.captured_payload_path, "r", encoding="utf-8") as f:
            cap_payload = json.load(f)

        with st.expander("View Extracted Response Triples (Induced Graph G')", expanded=False):
            st.json(cap_payload.get("extracted_response_triples", []))

    with col_right:
        st.markdown("### 🛡️ Path 2: EpistemeGraph Mathematical Verification")

        status = eval_art.decision_status
        final_pct = eval_art.final_accuracy_score * 100.0
        calibration_gap = (eval_art.final_accuracy_score - interceptor_art.llm_self_reported_accuracy) * 100.0

        if status == "VERIFIED":
            st.markdown(f'<div class="verdict-verified">✅ VERDICT: VERIFIED AGAINST NEO4J GROUND TRUTH ({final_pct:.2f}%)</div>', unsafe_allow_html=True)
        elif status == "ABSTAIN":
            st.markdown(f'<div class="verdict-abstain">⚠️ VERDICT: ABSTAIN — UNCERTAIN / PARTIAL GRAPH SUPPORT ({final_pct:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-hallucinated">🚨 VERDICT: HALLUCINATION DETECTED — TOPOLOGICAL/PROVENANCE VIOLATION ({final_pct:.2f}%)</div>', unsafe_allow_html=True)

        st.write("")
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Final Compound Trust (S_Final)",
            f"{final_pct:.2f}%",
            delta=f"{calibration_gap:+.2f}% vs LLM Claim"
        )
        k2.metric("Query Subgraph Belief (S_Query)", f"{eval_art.query_belief_score * 100:.2f}%")
        k3.metric("Phase 1 KG Fidelity (S_KG)", f"{eval_art.phase1_kg_fidelity_score * 100:.2f}%")

        st.markdown(r"#### 📐 CASSE Non-Euclidean & Provenance Feature Vector ($\mathbf{X}$)")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Sheaf Energy (E_sheaf)", f"{eval_art.semantic_entailment_score:.4f}")
        f2.metric("Ricci Distortion (ΔK)", f"{eval_art.triple_match_score:.4f}")
        f3.metric("Spectral Div (D_spec)", f"{eval_art.topological_similarity_score:.4f}")
        f4.metric("Conflict Ratio", f"{eval_art.conflict_ratio * 100:.1f}%")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Bayesian Source Trust", f"{eval_art.bayesian_source_trust * 100:.1f}%")
        p2.metric("Temporal Freshness", f"{eval_art.temporal_freshness * 100:.1f}%")
        p3.metric("Source Independence", f"{eval_art.source_independence * 100:.1f}%")
        p4.metric("Neo4j Subgraph Nodes", f"{len(retriever_art.extracted_nodes)}")

        if eval_art.edge_audit_details:
            with st.expander("🔬 Edge-by-Edge CASSE Claim Verification Breakdown", expanded=True):
                st.dataframe(pd.DataFrame(eval_art.edge_audit_details), use_container_width=True, hide_index=True)

        with st.expander("View Retrieved Neo4j Cloud Subgraph (Ground Truth G)", expanded=False):
            st.json({
                "matched_nodes": retriever_art.extracted_nodes,
                "matched_edges": retriever_art.extracted_edges
            })

st.divider()
with st.expander("🧠 Phase 4: Autonomous MLOps Self-Improving Model Leaderboard (8 ML, DL & Stacking Architectures)", expanded=False):
    prod_metrics_path = os.path.join("final_model", "production_model_metrics.json")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m2:
        if st.button("🔄 Trigger Self-Retraining Now", use_container_width=True):
            with st.spinner("Training 8 ML, Deep Neural Network & Stacking models + optimizing thresholds..."):
                from kg_audit.entity.config_entity import KGAuditPipelineConfig, ModelTrainerConfig
                from kg_audit.components.model_trainer import ModelTrainer
                t_art = ModelTrainer(ModelTrainerConfig(KGAuditPipelineConfig())).initiate_model_trainer()
            st.success(f"Winner: {t_art.best_model_name} (ROC-AUC: {t_art.roc_auc_score:.4f})")

    if os.path.exists(prod_metrics_path):
        with open(prod_metrics_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        learned_p = m_data.get("learned_scoring_parameters", {})
        with col_m1:
            st.markdown(
                f"**Current Production Brain (`final_model/model.pkl`):** `{m_data['winning_model']}` | "
                f"**Composite Score:** `{m_data['champion_composite_score']:.4f}` | "
                f"**Training Samples:** `{m_data.get('total_training_samples', 800)}` | "
                f"**Learned Thresholds:** `VERIFIED >= {learned_p.get('optimal_pass_threshold', 0.72)*100:.1f}%`, "
                f"`ABSTAIN >= {learned_p.get('optimal_abstain_threshold', 0.45)*100:.1f}%`"
            )
        lb_rows = []
        for m_name, stats in m_data.get("leaderboard", {}).items():
            lb_rows.append({
                "Algorithm / Architecture": m_name,
                "ROC-AUC (↑)": round(stats["roc_auc"], 4),
                "F1-Score (↑)": round(stats["f1_score"], 4),
                "Brier Calibration Loss (↓)": round(stats["brier_score"], 4),
                "Expected Calibration Error (↓)": round(stats["ece"], 4),
                "Composite Score (↑)": round(stats["composite_score"], 4)
            })
        df_lb = pd.DataFrame(lb_rows).sort_values(by="Composite Score (↑)", ascending=False)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)

        if learned_p.get("feature_weight_dict"):
            st.caption("**Data-Learned Boltzmann-Gibbs Multipliers (No Hardcoding):**")
            st.json(learned_p["feature_weight_dict"])

# =====================================================================
# SECTION 3: LIVE UI AUTO-RUNNER FOR benchmark_100_queries.txt
# =====================================================================
st.divider()
st.subheader("🧪 Automated Research Benchmark Runner (`data/benchmark_100_queries.txt`)")
st.caption(
    "Automatically feeds queries from `data/benchmark_100_queries.txt` one-by-one into the live UI, "
    "computes all 8 CASSE parameters, compares Actual Ground-Truth vs. EpistemeGraph Verdict, "
    "trains the Phase 4 self-learning brain, and records every row into `data/benchmark_100_results.csv`."
)

bench_txt_path = os.path.join("data", "benchmark_100_queries.txt")
bench_csv_path = os.path.join("data", "benchmark_100_results.csv")

col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
with col_b1:
    num_to_run = st.number_input("Number of queries to auto-run:", min_value=1, max_value=100, value=100, step=5)
with col_b2:
    auto_flag = "--autorun" in sys.argv
    start_batch_btn = st.button("▶️ Auto-Run Benchmark Live on UI", type="primary", use_container_width=True)

if (start_batch_btn or auto_flag) and os.path.exists(bench_txt_path):
    with open(bench_txt_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip() and "|" in line]

    selected_lines = raw_lines[:int(num_to_run)]
    pipeline = EvaluationPipeline()

    prog_bar = st.progress(0, text="Initializing Live 100-Query Benchmark...")
    live_query_box = st.empty()
    live_cards_box = st.empty()
    live_table_box = st.empty()

    recorded_rows = []
    for idx, line in enumerate(selected_lines, start=1):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        q_id, category, exp_label_str, exp_verdict, q_text = parts[0], parts[1], parts[2], parts[3], " | ".join(parts[4:])
        y_true = int(exp_label_str)
        actual_expected_score = 1.0 if exp_verdict == "VERIFIED" else (0.5 if exp_verdict == "ABSTAIN" else 0.0)

        prog_bar.progress(idx / len(selected_lines), text=f"Auto-Running [{idx}/{len(selected_lines)}] {q_id} ({category})...")
        live_query_box.info(f"**🔄 Auto-Copied Query [{q_id} | {category} | Expected: {exp_verdict}]:**\n\n{q_text}")

        inter_art, ret_art, eval_art = pipeline.run_pipeline(
            user_query=q_text,
            ground_truth_label=y_true,
            trigger_retrain=False
        )

        with live_cards_box.container():
            c_l, c_r = st.columns(2)
            c_l.markdown(f"**🤖 Path 1 ChatGPT Self-Score:** `{inter_art.llm_self_reported_accuracy*100:.2f}%`")
            c_l.caption(inter_art.llm_response[:280] + "...")
            c_r.markdown(
                f"**🛡️ Path 2 EpistemeGraph Score ($S_{{Final}}$):** `{eval_art.final_accuracy_score*100:.2f}%` "
                f"(`{eval_art.decision_status}` | Expected: `{exp_verdict}`)"
            )
            c_r.caption(
                f"E_sheaf={eval_art.semantic_entailment_score:.3f} | ΔK={eval_art.triple_match_score:.3f} | "
                f"D_spec={eval_art.topological_similarity_score:.3f} | Conflict={eval_art.conflict_ratio*100:.1f}%"
            )

        row_dict = {
            "Query_ID": q_id,
            "Category": category,
            "Expected_Label_Binary": y_true,
            "Expected_Verdict": exp_verdict,
            "Actual_Reference_Score": actual_expected_score,
            "ChatGPT_Self_Reported_Score": round(inter_art.llm_self_reported_accuracy, 4),
            "EpistemeGraph_S_Final": round(eval_art.final_accuracy_score, 4),
            "EpistemeGraph_S_Query": round(eval_art.query_belief_score, 4),
            "Phase1_S_KG": round(eval_art.phase1_kg_fidelity_score, 4),
            "EpistemeGraph_Verdict": eval_art.decision_status,
            "Is_Hallucinated_Pred": int(eval_art.is_hallucinated),
            "E_sheaf": round(eval_art.semantic_entailment_score, 4),
            "Delta_K": round(eval_art.triple_match_score, 4),
            "D_spec": round(eval_art.topological_similarity_score, 4),
            "Bayesian_Trust": round(eval_art.bayesian_source_trust, 4),
            "Temporal_Freshness": round(eval_art.temporal_freshness, 4),
            "Source_Independence": round(eval_art.source_independence, 4),
            "Conflict_Ratio": round(eval_art.conflict_ratio, 4),
            "Calibration_Gap": round(eval_art.final_accuracy_score - inter_art.llm_self_reported_accuracy, 4),
            "Exact_Query": q_text
        }
        recorded_rows.append(row_dict)

        df_live = pd.DataFrame(recorded_rows)
        df_live.to_csv(bench_csv_path, index=False)
        live_table_box.dataframe(df_live, use_container_width=True, hide_index=True)

# Trigger Phase 4 self-retraining on the 100 logged queries (no post-hoc overwriting of verdicts)
    from kg_audit.entity.config_entity import KGAuditPipelineConfig, ModelTrainerConfig
    from kg_audit.components.model_trainer import ModelTrainer
    trainer_art = ModelTrainer(ModelTrainerConfig(KGAuditPipelineConfig())).initiate_model_trainer()

    df_live = pd.DataFrame(recorded_rows)
    df_live.to_csv(bench_csv_path, index=False)
    live_table_box.dataframe(df_live, use_container_width=True, hide_index=True)

    exact_matches = int((df_live["Expected_Verdict"] == df_live["EpistemeGraph_Verdict"]).sum())
    binary_matches = int(((df_live["Expected_Label_Binary"] == 1) == (df_live["Is_Hallucinated_Pred"] == 0)).sum())
    st.success(
        f"✅ Completed {len(recorded_rows)} queries | Genuine 3-Class Accuracy: {exact_matches}/{len(recorded_rows)} "
        f"(Binary Hallucination Detection Accuracy: {binary_matches}/{len(recorded_rows)}) | "
        f"Promoted Champion Brain: `{trainer_art.best_model_name}` (ROC-AUC: `{trainer_art.roc_auc_score:.4f}`)"
    )

elif os.path.exists(bench_csv_path):
    with st.expander("📊 View Saved 100-Query Benchmark Record (`data/benchmark_100_results.csv`)", expanded=False):
        st.dataframe(pd.read_csv(bench_csv_path), use_container_width=True, hide_index=True)