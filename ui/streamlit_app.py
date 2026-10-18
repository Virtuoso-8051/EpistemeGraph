import os
import sys
import json
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

# Custom Dark Cyber-Research Styling
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
    "Dual-Path Middleware: Phase 1 Multi-PDF Knowledge Graph Fidelity ($S_{KG}$) "
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
    st.write("Upload multiple PDFs from your laptop (`paper1.pdf`, `paper2.pdf`, etc.) to build & audit the Ground-Truth Knowledge Graph.")

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

    build_btn = st.button("🚀 Construct KG & Push to Neo4j Cloud", use_container_width=True, type="primary")

    if build_btn:
        target_files = saved_file_paths if saved_file_paths else existing_files
        if not target_files:
            st.error("Please upload at least one PDF or place files in `data/uploads/`!")
        else:
            with st.spinner("Chunking PDFs, extracting triplets via OpenAI, resolving cross-PDF entities, and pushing to Neo4j AuraDB Cloud..."):
                pipeline = EvaluationPipeline()
                kg_artifact = pipeline.run_phase1_kg_construction(file_paths=target_files)
                st.session_state["latest_kg_artifact"] = kg_artifact
                st.session_state["latest_kg_json_path"] = kg_artifact.extracted_kg_file_path
            st.success("Phase 1 Complete! KG stored locally & pushed to Neo4j Cloud.")

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
            st.markdown(f"**🔗 Entities Linked Across Multiple PDFs:** `{', '.join(shared_list[:20])}`")

        # Render Interactive PyVis Graph if available
        latest_kg_path = st.session_state.get("latest_kg_json_path")
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
                rel = edge.get("relation", "RELATED_TO")
                quote = edge.get("evidence_quote", "")
                if u and v:
                    net.add_edge(u, v, label=rel, title=f"[{edge.get('source_name')}] {quote}", color="#64748b")

            html_path = os.path.join("Artifacts", "pyvis_kg.html")
            os.makedirs("Artifacts", exist_ok=True)
            net.save_graph(html_path)
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
    placeholder="e.g., How do the concepts in paper 1 connect to the methodology in paper 2?",
    height=90
)

verify_btn = st.button("🔍 Intercept Query & Run CASSE Verification", type="primary")

if verify_btn and user_query.strip():
    with st.spinner("Path 1: Querying OpenAI & capturing self-reported confidence | Path 2: Pulling Neo4j Cloud Subgraph & computing CASSE Tensor..."):
        pipeline = EvaluationPipeline()
        interceptor_art, retriever_art, eval_art = pipeline.run_pipeline(user_query=user_query.strip())

    col_left, col_right = st.columns(2, gap="large")

    # ------------------------------------------------------------------
    # LEFT COLUMN: PATH 1 (RAW LLM OUTPUT & SELF-REPORTED ACCURACY)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # RIGHT COLUMN: PATH 2 (EPISTEMEGRAPH CASSE + ML AUDIT)
    # ------------------------------------------------------------------
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

        st.markdown("#### 📐 CASSE Non-Euclidean & Provenance Feature Vector ($\mathbf{X}$)")
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

        with st.expander("View Retrieved Neo4j Cloud Subgraph (Ground Truth G)", expanded=False):
            st.json({
                "matched_nodes": retriever_art.extracted_nodes,
                "matched_edges": retriever_art.extracted_edges
            })