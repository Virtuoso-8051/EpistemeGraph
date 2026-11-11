r"""
Export every number, table and figure used by paper/main.tex from the actual run artefacts.

  paper/generated/results.tex   \newcommand macros + LaTeX tables (\input by main.tex)
  paper/figures/*.pdf           reliability diagram, ROC curve, permutation importance

Nothing in the manuscript is typed by hand: if an artefact is missing, the macro renders as a
visible placeholder ("--") so an un-run result can never be mistaken for a real one.

Usage: python scripts/export_paper_assets.py [--workspace <id>]
  -> also writes paper/EpistemeGraph_overleaf.zip (upload to Overleaf, compiles in one go)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kg_audit.components.decision import FEATURE_SCHEMA_VERSION, FEATURES  # noqa: E402
from kg_audit.utils.main_utils import read_json  # noqa: E402

GEN = os.path.join(ROOT, "paper", "generated")
FIG = os.path.join(ROOT, "paper", "figures")
DASH = "--"


def f(x: Any, nd: int = 3, pct: bool = False) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return DASH
        return f"{100 * v:.1f}" if pct else f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return DASH


def esc(s: Any) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


MODEL_NAMES = {"MonotoneHistGBM": "Monotone gradient boosting", "GradientBoosting": "Gradient boosting",
               "StackingEnsemble": "Stacking ensemble", "RandomForest": "Random forest", "RBF_SVM": "RBF-kernel SVM",
               "ExtraTrees": "Extra trees", "LogisticRegression": "Logistic regression", "AttentionMIL": "Attention MIL network",
               "MLP": "Multilayer perceptron", "AttentionResNet": "Attention ResNet", "TabPFN": "TabPFN"}
CLAIM_NAMES = {"s_prov": "Provenance strength", "tier_direct": "Direct KG edge", "tier_two_hop": "Two-hop KG path",
               "conflict": "Conflict flag", "pol_asserted": "Asserted polarity", "pol_speculative": "Speculative polarity",
               "pol_refuted": "Refuted polarity", "aligned_frac": "Entity alignment", "nli_entail": "NLI entailment",
               "nli_contra": "NLI contradiction", "nli_cons": "NLI consistency", "cos_evidence": "Evidence similarity",
               "ooc_rate": "Out-of-corpus rate", "sheaf_resid": "Sheaf residual", "forman_graft": "Grafted-edge curvature"}
ABL_NAMES = {"all_features": "All inputs", "answer_features_only_v3": "13 answer features only", "pooled_claims_only": "Pooled claims only",
             "topological_only": "Topological only", "evidential_only": "Evidential only", "model_signals_only": "Model signals only",
             "text_nli_only": "Text NLI only", "graph_only": "Graph only", "without_topological": "Without topological",
             "without_model_signals": "Without model signals"}


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


TABLE_TEMPLATE = r"""\newcommand{\NAME}{%
\begin{table}[t]\centering\footnotesize
\caption{CAPTION}\label{LABEL}
\begin{adjustbox}{max width=\columnwidth}\begin{tabular}{ALIGN}\toprule
HEADER \\ \midrule
BODY
\bottomrule\end{tabular}\end{adjustbox}
\end{table}}
"""


def table(caption: str, label: str, header: List[str], rows: List[List[str]], align: str) -> str:
    """A float that only SHRINKS to the column width when too wide (never blows small tables up); a table with no
    data yet expands to nothing instead of printing placeholders."""
    name = "Table" + label.split(":")[1].capitalize()
    if not rows:
        return "\\newcommand{\\" + name + "}{}\n"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (TABLE_TEMPLATE.replace("NAME", name).replace("CAPTION", caption).replace("LABEL", label)
            .replace("ALIGN", align).replace("HEADER", " & ".join(header)).replace("BODY", body))


def figures(pred: Optional[pd.DataFrame], prod: Dict[str, Any]) -> List[str]:
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return made
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    if pred is not None and len(pred) and pred["label"].nunique() == 2:
        from sklearn.metrics import roc_curve
        # reliability diagram
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        bins = np.linspace(0, 1, 11)
        for col, lab, mk in (("s_final", r"EpistemeGraph $S_{\mathrm{final}}$", "o"), ("c_llm", r"LLM self-confidence $c_{\mathrm{LLM}}$", "s")):
            p = pred[col].clip(0, 1).to_numpy()
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [pred["label"].to_numpy()[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax.plot(xs, ys, marker=mk, ms=4, lw=1.2, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        ax.set_xlabel("predicted P(supported)")
        ax.set_ylabel("empirical frequency")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "reliability.pdf"))
        plt.close(fig)
        made.append("reliability.pdf")
        # ROC
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        for col, lab in (("s_final", r"$S_{\mathrm{final}}$"), ("s_query", r"$S_{\mathrm{query}}$"), ("c_llm", r"$c_{\mathrm{LLM}}$")):
            fpr, tpr, _ = roc_curve(pred["label"], pred[col])
            ax.plot(fpr, tpr, lw=1.2, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "roc.pdf"))
        plt.close(fig)
        made.append("roc.pdf")
    imp = prod.get("permutation_importance")
    if imp:
        keys = [ft["key"] for ft in FEATURES]
        m = np.array([imp[k]["mean"] for k in keys])
        s = np.array([imp[k]["std"] for k in keys])
        order = np.argsort(m)
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        ax.barh(range(len(keys)), m[order], xerr=s[order], color="#4C72B0", ecolor="grey", capsize=2)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f"${FEATURES[i]['tex']}$" for i in order])
        ax.set_xlabel(r"$\Delta$ log-loss when permuted")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "importance.pdf"))
        plt.close(fig)
        made.append("importance.pdf")
    return made


def overleaf_bundle() -> str:
    """paper/EpistemeGraph_overleaf.zip: upload to Overleaf (New Project -> Upload Project) and press Recompile."""
    import zipfile
    out = os.path.join(ROOT, "paper", "EpistemeGraph_overleaf.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ("main.tex", "extended.tex", "references.bib", os.path.join("generated", "results.tex"), os.path.join("generated", "defaults.tex")):
            p = os.path.join(ROOT, "paper", rel)
            if os.path.exists(p):
                z.write(p, rel.replace(os.sep, "/"))
        if os.path.isdir(FIG):
            for fn in sorted(os.listdir(FIG)):
                if fn.endswith(".pdf"):
                    z.write(os.path.join(FIG, fn), f"figures/{fn}")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None, help="workspace id (default: active)")
    args = ap.parse_args(argv or [])
    os.chdir(ROOT)
    from kg_audit.utils import workspace as wsm
    W = wsm.workspace_dir(args.workspace or wsm.active_workspace_id())

    os.makedirs(GEN, exist_ok=True)
    kg = read_json(os.path.join(W, "kg_fidelity_state.json"), default={}) or {}
    prod = read_json(os.path.join(W, "production_model_metrics.json"), default={}) or {}
    if prod.get("schema_version") != FEATURE_SCHEMA_VERSION:
        prod = {}
    agree = read_json(os.path.join(W, "benchmark", "judge_agreement.json"), default={}) or {}
    xjudge = read_json(os.path.join(W, "benchmark", "judge_crosscheck.json"), default={}) or {}
    ana = read_json(os.path.join(W, "benchmark", "results_analysis.json"), default={}) or {}
    v3 = read_json(os.path.join(W, "brain_v3_backup", "production_model_metrics.json"), default={}) or {}
    hk = read_json(os.path.join(W, "brain_v4_haikulabels", "production_model_metrics.json"), default={}) or {}
    hk_ana = read_json(os.path.join(W, "brain_v4_haikulabels", "results_analysis.json"), default={}) or {}
    third = read_json(os.path.join(W, "benchmark", "judge_third.json"), default={}) or {}
    mtc = ana.get("matching") or {}

    def sentence_third() -> str:
        if not third:
            return ""
        ag = (third.get("agreement") or {}).get(third.get("primary_judge"), {})
        return (r"A third judge from a different model family (\texttt{" + esc(third.get("third_judge", "")) + r"}), given the "
                r"three papers in full and blind to the other verdicts, labelled the same " + str(third.get("n")) +
                r" answers; its binary agreement with the primary judge is $\kappa=" + f(ag.get("kappa_binary"), nd=2) +
                r"$ (raw agreement " + f(ag.get("raw_agreement_binary"), pct=True) + r"\%).")

    def sentence_human() -> str:
        if not agree:
            return ""
        return (r"On " + str(agree.get("n")) + r" answers annotated by the author, binary human--judge agreement is "
                r"Cohen's $\kappa=" + f(agree.get("cohen_kappa_binary"), nd=2) + r"$.")
    cvm = prod.get("cv_metrics") or {}
    cvmod = cvm.get("models") or {}
    run_cfg = read_json(os.path.join(W, "benchmark", "runs", "latest", "config.json"), default={}) or {}
    kga = read_json(os.path.join(W, "kg_precision_audit.json"), default={}) or {}
    sens = read_json(os.path.join(W, "sensitivity_phase1.json"), default={}) or {}
    ma = ((kga.get("human") or kga.get("llm") or {}).get("model_assisted") or {}) if kga else {}
    if not ma.get("available"):
        ma = {}
    kg_src = kga.get("human") or kga.get("llm") or {}
    kg_p = (kg_src.get("overall") or {}).get("strict", {})
    kg_pl = (kg_src.get("overall") or {}).get("lenient", {})
    pred_path = os.path.join(W, "test_predictions.csv")
    pred = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
    res_path = os.path.join(W, "benchmark", "results.csv")
    res = pd.read_csv(res_path) if os.path.exists(res_path) else None
    if res is not None and "S_final" not in res.columns:
        res = None

    t = prod.get("test_metrics", {})
    sf, conf, base = t.get("S_final", {}), t.get("conformal", {}), t.get("baselines", {})
    pool = (prod.get("decision_bundle") or {}).get("pool", {})
    cb = (prod.get("decision_bundle") or {}).get("conformal", {})
    sizes = prod.get("split_sizes", {})
    sig = t.get("significance", {}) or {}

    def pv(x):
        try:
            return r"{<}\,0.001" if float(x) < 0.001 else f"{float(x):.3f}"     # math-mode safe: used as $p=\Macro$
        except (TypeError, ValueError):
            return DASH

    # claim-level hallucination extent vs judge label (validates the "how much" measure)
    ext_hal = ext_sup = ext_auc = None
    if res is not None and "hallucination_extent" in res.columns and res["label"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        ext_hal = res.loc[res["label"] == 0, "hallucination_extent"].mean()
        ext_sup = res.loc[res["label"] == 1, "hallucination_extent"].mean()
        ext_auc = roc_auc_score(res["label"], 1.0 - res["hallucination_extent"])

    m = [
        "% AUTO-GENERATED by scripts/export_paper_assets.py -- do not edit by hand",
        macro("SKG", f(kg.get("overall_kg_accuracy_score"), pct=True)),
        macro("PhiProv", f(kg.get("provenance_grounding_score"), pct=True)),
        macro("PhiCov", f(kg.get("semantic_coverage_score"), pct=True)),
        macro("PhiTopo", f(kg.get("topological_coherence_score"), pct=True)),
        macro("PhiCons", f(kg.get("internal_consistency_score"), pct=True)),
        macro("NDocs", str(kg.get("documents_in_corpus", DASH))),
        macro("NChunks", str(kg.get("total_chunks_processed", DASH))),
        macro("NRawTriples", str(kg.get("raw_triplets_extracted", DASH))),
        macro("NTriples", str(kg.get("verified_triplets_pushed", DASH))),
        macro("NEntities", str(kg.get("resolved_entity_count", DASH))),
        macro("NShared", str(kg.get("cross_pdf_shared_entities_count", DASH))),
        macro("AcceptRate", f((kg.get("diagnostics") or {}).get("claim_acceptance_rate"), pct=True)),
        macro("KGPrec", f(kg_p.get("p"), pct=True)), macro("KGPrecLo", f(kg_p.get("lo"), pct=True)),
        macro("KGPrecHi", f(kg_p.get("hi"), pct=True)), macro("KGPrecLenient", f(kg_pl.get("p"), pct=True)),
        macro("KGAuditN", str(kg_p.get("n", DASH))),
        macro("KGAuditor", "human annotators" if kga.get("human") else (esc(kga.get("auditor_model")) if kga else DASH)),
        macro("NItems", str(len(res)) if res is not None else DASH),
        macro("NLabelled", str(prod.get("n_labelled", DASH))),
        macro("NTrain", str(sizes.get("train", DASH))), macro("NCal", str(sizes.get("cal", DASH))),
        macro("NTest", str(sizes.get("test", DASH))),
        macro("SupportedRate", f(res["label"].mean(), pct=True) if res is not None else DASH),
        macro("ExtentHal", f(ext_hal, pct=True)), macro("ExtentSup", f(ext_sup, pct=True)),
        macro("ExtentAUC", f(ext_auc)),
        macro("Champion", esc(MODEL_NAMES.get(prod.get("winning_model", ""), prod.get("winning_model", DASH))).lower()),
        macro("TestAUC", f(sf.get("roc_auc"))), macro("TestLL", f(sf.get("log_loss"))),
        macro("TestBrier", f(sf.get("brier"))), macro("TestECE", f(sf.get("ece"))),
        macro("TestAcc", f(sf.get("accuracy_at_0.5"), pct=True)),
        macro("LLMAUC", f((base.get("llm_self_confidence") or {}).get("roc_auc"))),
        macro("LLMECE", f((base.get("llm_self_confidence") or {}).get("ece"))),
        macro("Alpha", f(cb.get("alpha"), nd=2)),
        macro("Coverage", f(conf.get("coverage"), pct=True)),
        macro("CoverageSup", f(conf.get("coverage_supported"), pct=True)),
        macro("CoverageHal", f(conf.get("coverage_hallucinated"), pct=True)),
        macro("AbstainRate", f(conf.get("abstention_rate"), pct=True)),
        macro("SelAcc", f(conf.get("selective_accuracy"), pct=True)),
        macro("TauVer", f(cb.get("tau_verified"))), macro("TauHal", f(cb.get("tau_hallucinated"))),
        macro("PoolBzero", f(pool.get("b0"), nd=2)), macro("PoolBq", f(pool.get("b_query"), nd=2)),
        macro("PoolBkg", f(pool.get("b_kg"), nd=2)),
        macro("JudgeKappa", f(agree.get("cohen_kappa_binary"), nd=2)),
        macro("JudgeN", str(agree.get("n", DASH))),
        macro("PrimaryJudge", str(xjudge.get("primary_judge", DASH))),
        macro("CrossJudge", str(xjudge.get("second_judge", DASH))),
        macro("CrossJudgeN", str(xjudge.get("n", DASH))),
        macro("CrossJudgeKappa", f(xjudge.get("cohen_kappa_binary"), nd=2)),
        macro("CrossJudgeAgree", f(100 * xjudge["raw_agreement_binary"], nd=1) if xjudge.get("raw_agreement_binary") is not None else DASH),
        macro("TestAUCLo", f((t.get("S_final_ci95") or {}).get("roc_auc", [None, None])[0])),
        macro("TestAUCHi", f((t.get("S_final_ci95") or {}).get("roc_auc", [None, None])[1])),
        macro("LLMAUCLo", f(((base.get("llm_self_confidence") or {}).get("ci95") or {}).get("roc_auc", [None, None])[0])),
        macro("LLMAUCHi", f(((base.get("llm_self_confidence") or {}).get("ci95") or {}).get("roc_auc", [None, None])[1])),
        macro("DeltaLLM", f((sig.get("vs_llm_self_confidence") or {}).get("delta"))),
        macro("PvalLLM", pv((sig.get("vs_llm_self_confidence") or {}).get("p_value"))),
        macro("DeltaText", f((sig.get("vs_text_nli_only") or {}).get("delta"))),
        macro("PvalText", pv((sig.get("vs_text_nli_only") or {}).get("p_value"))),
        macro("DeltaTopo", f((sig.get("all_vs_without_topological") or {}).get("delta"))),
        macro("PvalTopo", pv((sig.get("all_vs_without_topological") or {}).get("p_value"))),
        macro("KGAccLearned", f(ma.get("estimate"), pct=True)), macro("KGAccLo", f(ma.get("lo"), pct=True)),
        macro("KGAccHi", f(ma.get("hi"), pct=True)),
        macro("SensDefault", f(sens.get("S_KG_default"), nd=4)), macro("SensMaxDev", f(sens.get("max_abs_deviation"), nd=4)),
        macro("SensRange", f(sens.get("S_KG_range"), nd=4)),
        # ---- v4: models, repeated CV over all labelled answers, judge reliability, cost
        macro("AnswerModel", esc(run_cfg.get("answer_model", DASH))),
        macro("JudgeModel", esc(run_cfg.get("judge_model", DASH))),
        macro("CVProtocol", (r"5$\times$5 repeated stratified cross-validation over all " + str(cvm["n"]) + " answers") if cvm.get("n") else DASH),
        macro("CVAUC", f((cvmod.get("EpistemeGraph_v4") or {}).get("auroc_pooled_oof"))),
        macro("CVAUCsd", f((cvmod.get("EpistemeGraph_v4") or {}).get("auroc_sd_folds"))),
        macro("CVAUCLo", f(((cvmod.get("EpistemeGraph_v4") or {}).get("auroc_ci95_rep0") or [None, None])[0])),
        macro("CVAUCHi", f(((cvmod.get("EpistemeGraph_v4") or {}).get("auroc_ci95_rep0") or [None, None])[1])),
        macro("CVAUCvthree", f((cvmod.get("same_learner_v3_inputs") or {}).get("auroc_pooled_oof"))),
        macro("CVAUCtext", f((cvmod.get("text_nli_checker") or {}).get("auroc_pooled_oof"))),
        macro("CVAUCllm", f((cvmod.get("llm_self_confidence") or {}).get("auroc_pooled_oof"))),
        macro("CVAUClen", f((cvmod.get("answer_length") or {}).get("auroc_pooled_oof"))),
        macro("CVECE", f((cvmod.get("EpistemeGraph_v4") or {}).get("ece_rep0"))),
        macro("CVECEllm", f((cvmod.get("llm_self_confidence") or {}).get("ece_rep0"))),
        macro("CVPvthree", pv(((cvm.get("delong_rep0") or {}).get("v4_vs_same_learner_v3_inputs") or {}).get("p_value"))),
        macro("CVPllm", pv(((cvm.get("delong_rep0") or {}).get("v4_vs_llm_self_confidence") or {}).get("p_value"))),
        macro("CVPtext", pv(((cvm.get("delong_rep0") or {}).get("v4_vs_text_nli_checker") or {}).get("p_value"))),
        macro("CVPlen", pv(((cvm.get("delong_rep0") or {}).get("v4_vs_answer_length") or {}).get("p_value"))),
        macro("SevereAUC", f((cvm.get("severe_hallucination_target") or {}).get("auroc"))),
        macro("SevereLLM", f((cvm.get("severe_hallucination_target") or {}).get("llm_self_confidence_auroc"))),
        macro("SevereN", str((cvm.get("severe_hallucination_target") or {}).get("positives", DASH))),
        macro("VthreeTestAUC", f(((v3.get("test_metrics") or {}).get("S_final") or {}).get("roc_auc"))),
        macro("VthreeChampion", esc(v3.get("winning_model", DASH))),
        macro("KappaUnsup", f((((ana.get("judge_agreement") or {}).get("kappa") or {}).get("unsupported_vs_rest") or {}).get("kappa"), nd=2)),
        macro("KappaOrd", f((((ana.get("judge_agreement") or {}).get("kappa") or {}).get("ordinal_linear_weighted") or {}).get("kappa"), nd=2)),
        macro("AUCvsSecond", f((ana.get("vs_second_judge_labels") or {}).get("auroc_v4_vs_second_judge"))),
        macro("AUCvsPrimarySame", f((ana.get("vs_second_judge_labels") or {}).get("auroc_v4_vs_primary_judge_same_items"))),
        macro("AUCagree", f((ana.get("vs_second_judge_labels") or {}).get("auroc_v4_on_items_where_judges_agree"))),
        macro("NAgree", str((ana.get("vs_second_judge_labels") or {}).get("n_agree", DASH))),
        macro("LenAUC", f((ana.get("confounds") or {}).get("answer_length_auroc_for_supported"))),
        macro("FPAccepted", f(ana.get("false_premise_accepted_rate"), pct=True)),
        macro("NPartial", str((ana.get("verdicts") or {}).get("PARTIALLY_SUPPORTED", DASH))),
        macro("NUnsup", str((ana.get("verdicts") or {}).get("UNSUPPORTED", DASH))),
        macro("NSup", str((ana.get("verdicts") or {}).get("SUPPORTED", DASH))),
        macro("CostRun", f((ana.get("cost") or {}).get("ledger_total_usd"), nd=2)),
        macro("KappaFull", f((ana.get("judge_agreement_full") or {}).get("kappa_binary"), nd=2)),
        macro("AgreeFull", f((ana.get("judge_agreement_full") or {}).get("raw_agreement_binary"), pct=True)),
        macro("ConsensusN", str((ana.get("consensus_two_judges") or {}).get("n", DASH))),
        macro("ConsensusAUC", f((ana.get("consensus_two_judges") or {}).get("auroc_v4"))),
        macro("ConsensusLLM", f((ana.get("consensus_two_judges") or {}).get("auroc_llm"))),
        macro("ConsensusMatched", str((ana.get("consensus_two_judges") or {}).get("matched_at_0.5", DASH))),
        macro("GeminiName", esc((ana.get("third_judge") or {}).get("judge", DASH))),
        macro("GeminiSupp", f((ana.get("third_judge") or {}).get("supported_rate"), pct=True)),
        macro("GeminiKappa", f(((((ana.get("third_judge") or {}).get("agreement") or {}).get(run_cfg.get("judge_model", "")) or {}).get("kappa_binary")), nd=2)),
        macro("GeminiAUC", f((ana.get("third_judge") or {}).get("auroc_v4_vs_third"))),
        macro("GeminiLLMAUC", f((ana.get("third_judge") or {}).get("auroc_llm_vs_third"))),
        macro("MajorityAUC", f((ana.get("third_judge") or {}).get("auroc_v4_vs_majority"))),
        macro("MajorityLLMAUC", f((ana.get("third_judge") or {}).get("auroc_llm_vs_majority"))),
        macro("NUnanimous", str((ana.get("third_judge") or {}).get("n_unanimous", DASH))),
        macro("ThirdJudgeSentence", sentence_third()), macro("HumanJudgeSentence", sentence_human()),
        macro("HaikuLabelCVAUC", f((((hk.get("cv_metrics") or {}).get("models") or {}).get("EpistemeGraph_v4") or {}).get("auroc_pooled_oof"))),
        macro("HaikuLabelCVAUCllm", f((((hk.get("cv_metrics") or {}).get("models") or {}).get("llm_self_confidence") or {}).get("auroc_pooled_oof"))),
        macro("HaikuLabelCVAUCvthree", f((((hk.get("cv_metrics") or {}).get("models") or {}).get("same_learner_v3_inputs") or {}).get("auroc_pooled_oof"))),
        macro("CatCrossLLM", f(((ana.get("oof_auroc_by_category") or {}).get("CROSS_DOC") or {}).get("llm_self_confidence"))),
        macro("CatFPLLM", f(((ana.get("oof_auroc_by_category") or {}).get("FALSE_PREMISE") or {}).get("llm_self_confidence"))),
        macro("HKvsSonnet", f((hk_ana.get("vs_second_judge_labels") or {}).get("auroc_v4_vs_second_judge"))),
        macro("HKvsHaiku", f((hk_ana.get("vs_second_judge_labels") or {}).get("auroc_v4_vs_primary_judge_same_items"))),
        macro("HaikuLabelKappa", f((((hk_ana.get("judge_agreement") or {}).get("kappa") or {}).get("supported_vs_rest") or {}).get("kappa"), nd=2)),
        macro("MatchN", str(mtc.get("matched_at_0.5", DASH))), macro("MatchTotal", str(mtc.get("n", DASH))),
        macro("MatchBase", str(mtc.get("majority_baseline", DASH))), macro("BalAcc", f(mtc.get("balanced_accuracy"), pct=True)),
        macro("Committed", str(mtc.get("committed", DASH))), macro("CommittedCorrect", str(mtc.get("committed_correct", DASH))),
        macro("KGPrecLenientLo", f(kg_pl.get("lo"), pct=True)), macro("KGPrecLenientHi", f(kg_pl.get("hi"), pct=True)),
        macro("KGNIncorrect", str(sum(1 for i in (kga.get("items") or []) if i.get("llm_verdict") == "INCORRECT")) if kga else DASH),
        macro("KGNPartial", str(sum(1 for i in (kga.get("items") or []) if i.get("llm_verdict") == "PARTIALLY_CORRECT")) if kga else DASH),
        macro("CVCovTen", f(next((c["coverage"] for c in ana.get("conformal_curve_cv", []) if abs(c["alpha"] - 0.1) < 1e-9), None), pct=True)),
        macro("CVAbsTen", f(next((c["abstention_rate"] for c in ana.get("conformal_curve_cv", []) if abs(c["alpha"] - 0.1) < 1e-9), None), pct=True)),
        macro("CVSelTen", f(next((c["selective_accuracy"] for c in ana.get("conformal_curve_cv", []) if abs(c["alpha"] - 0.1) < 1e-9), None), pct=True)),
        macro("CVAbsTwenty", f(next((c["abstention_rate"] for c in ana.get("conformal_curve_cv", []) if abs(c["alpha"] - 0.2) < 1e-9), None), pct=True)),
        macro("CVSelTwenty", f(next((c["selective_accuracy"] for c in ana.get("conformal_curve_cv", []) if abs(c["alpha"] - 0.2) < 1e-9), None), pct=True)),
        macro("PoolLRTp", f(pool.get("lrt_p"))), macro("PoolLRTchi", f(pool.get("lrt_chi2"), nd=2)),
        macro("PoolKGused", "kept" if pool.get("kg_term_used") else "dropped"),
        macro("CatBorderLLM", f(((ana.get("oof_auroc_by_category") or {}).get("BORDERLINE") or {}).get("llm_self_confidence"))),
        macro("CatBorderLen", f(((ana.get("oof_auroc_by_category") or {}).get("BORDERLINE") or {}).get("answer_length"))),
        macro("CatBorderVfour", f(((ana.get("oof_auroc_by_category") or {}).get("BORDERLINE") or {}).get("EpistemeGraph_v4"))),
        macro("CatCrossVfour", f(((ana.get("oof_auroc_by_category") or {}).get("CROSS_DOC") or {}).get("EpistemeGraph_v4"))),
        macro("CatCrossVthree", f(((ana.get("oof_auroc_by_category") or {}).get("CROSS_DOC") or {}).get("same_learner_v3_inputs"))),
        macro("CatFactLLM", f(((ana.get("oof_auroc_by_category") or {}).get("FACTUAL") or {}).get("llm_self_confidence"))),
        macro("CatFactVfour", f(((ana.get("oof_auroc_by_category") or {}).get("FACTUAL") or {}).get("EpistemeGraph_v4"))),
        macro("CatFPVfour", f(((ana.get("oof_auroc_by_category") or {}).get("FALSE_PREMISE") or {}).get("EpistemeGraph_v4"))),
        macro("PoolCVwith", f(pool.get("cv_logloss_with_kg"), nd=4)), macro("PoolCVwithout", f(pool.get("cv_logloss_without_kg"), nd=4)),
        macro("TestAUCq", f((t.get("S_query") or {}).get("roc_auc"))),
        macro("AUCvsSecondLLM", f((ana.get("vs_second_judge_labels") or {}).get("auroc_llm_conf_vs_second_judge"))),
        macro("OrdAUC", f((ana.get("label_design") or {}).get("ordinal_auroc"))),
        macro("BinAUC", f((ana.get("label_design") or {}).get("binary_auroc"))),
    ]

    lb_rows = []
    for name, s in sorted((prod.get("leaderboard") or {}).items(), key=lambda kv: kv[1]["log_loss"]):
        star = r"$^\star$" if name == prod.get("winning_model") else ""
        lb_rows.append([esc(MODEL_NAMES.get(name, name)) + star, f(s["log_loss"]), f(s["roc_auc"]), f(s["brier"]), f(s["ece"])])
    abl_rows = [[esc(ABL_NAMES.get(k, k.replace("_", " "))), f(v["roc_auc"]),
                 f"[{f((v.get('ci95') or {}).get('roc_auc', [None, None])[0])}, {f((v.get('ci95') or {}).get('roc_auc', [None, None])[1])}]",
                 f(v["log_loss"]), f(v["ece"])] for k, v in (t.get("ablations") or {}).items()]
    sens_rows = [[("all defaults" if r["parameter"] == "default" else esc(r["meaning"])), DASH if r["value"] is None else str(r["value"]),
                  f(r["S_KG"], nd=4), f(r["phi_prov"]), f(r["phi_cov"]), f(r["phi_topo"]), str(r["verified"]), str(r["entities"])]
                 for r in sens.get("rows", [])]
    tw_rows = [[esc(k.replace("_", " ")), f(v, nd=2)] for k, v in sorted((ma.get("learned_weights") or {}).items(), key=lambda kv: -kv[1])]
    w = (prod.get("learned_feature_weights") or {}).get("standardised_coefficients", {})
    imp = prod.get("permutation_importance") or {}
    feat_rows = []
    if prod:
        for ft in FEATURES:
            k = ft["key"]
            feat_rows.append([f"${ft['tex']}$", ft["name"], ft["family"],
                              f((base.get(f'single::{k}') or {}).get("roc_auc")), f(w.get(k), nd=2),
                              f((imp.get(k) or {}).get("mean"), nd=3)])
    cat_rows = []
    for cat, s in (t.get("per_category") or {}).items():
        d = s.get("decisions", {})
        cat_rows.append([esc(cat), str(s["n"]), f(s["label_rate"], pct=True), f(s["mean_S_final"]),
                         str(d.get("VERIFIED", 0)), str(d.get("ABSTAIN", 0)), str(d.get("HALLUCINATED", 0))])
    kg_rows = []
    for pdf, s in sorted((kg_src.get("by_pdf") or {}).items()):
        st = s["strict"]
        kg_rows.append([esc(pdf), str(st["n"]), f(st["p"], pct=True), f"[{f(st['lo'], pct=True)}, {f(st['hi'], pct=True)}]",
                        f(s["lenient"]["p"], pct=True)])

    cv_rows = []
    dl = cvm.get("delong_rep0") or {}
    for k, lab in (("EpistemeGraph_v4", r"\textbf{EpistemeGraph}"),
                   ("same_learner_v3_inputs", "Same learner, 13 answer features"),
                   ("text_nli_checker", "Text-only NLI checker"),
                   ("llm_self_confidence", "LLM verbalised confidence"), ("answer_length", "Answer length")):
        v = cvmod.get(k)
        if not v:
            continue
        ci = v.get("auroc_ci95_rep0") or [None, None]
        cv_rows.append([lab, f(v["auroc_pooled_oof"]), f"{f(v['auroc_mean'])} $\\pm$ {f(v['auroc_sd_folds'])}",
                        f"[{f(ci[0])}, {f(ci[1])}]", f(v.get("ece_rep0")),
                        DASH if k == "EpistemeGraph_v4" else "$" + pv((dl.get(f"v4_vs_{k}") or {}).get("p_value")) + "$"])
    impc = prod.get("permutation_importance_claims") or {}
    ci_rows = [[esc(CLAIM_NAMES.get(k, k.replace("_", " "))), f(v["mean"], nd=4), f(v["std"], nd=4)]
               for k, v in sorted(impc.items(), key=lambda kv: -kv[1]["mean"])]

    cc_rows = [[f(c["alpha"], nd=2), f(c["coverage"], pct=True), f(c["coverage_supported"], pct=True), f(c["coverage_hallucinated"], pct=True),
                f(c["abstention_rate"], pct=True), f(c["selective_accuracy"], pct=True)] for c in ana.get("conformal_curve_cv", [])]
    tables = [
        table(r"Repeated cross-validation over all labelled answers (\CVProtocol). AUROC of the pooled out-of-fold scores averaged over the five "
              r"repetitions, mean $\pm$ s.d.\ over the 25 folds, 95\% bootstrap CI and ECE of the first repetition, and the "
              r"paired DeLong $p$-value against EpistemeGraph.", "tab:cv",
              ["Detector", "AUROC", "fold mean $\\pm$ s.d.", "95\\% CI", "ECE", "$p$ (DeLong)"], cv_rows, "lccccc"),
        table(r"Risk--abstention trade-off of the Mondrian cross-conformal decision over all labelled answers "
              r"(out-of-fold scores, 5-fold calibration).", "tab:conformal",
              [r"$\alpha$", "Coverage", r"Cov.\ sup.", r"Cov.\ hal.", "Abstain", r"Sel.\ acc."], cc_rows, "cccccc"),
        table("Grouped permutation importance of the pooled claim statistics (mean, min, max of one claim feature "
              "permuted jointly; increase in test log-loss).", "tab:claimimp", ["Claim feature", "$\\Delta$ log-loss", "s.d."], ci_rows, "lcc"),
        table(r"Ten-model tournament (nested 5$\times$3 CV on the training split, out-of-fold metrics). "
              r"$^\star$ marks the champion, selected by log-loss.", "tab:leaderboard",
              ["Model", "Log-loss$\\downarrow$", "AUROC$\\uparrow$", "Brier$\\downarrow$", "ECE$\\downarrow$"], lb_rows, "lcccc"),
        table("Feature-family ablation (standardised logistic model, held-out test split; 95\\% bootstrap CI).", "tab:ablation",
              ["Feature set", "AUROC", "95\\% CI", "Log-loss", "ECE"], abl_rows, "lcccc"),
        table("Phase 1 robustness: every design constant perturbed one at a time and the pipeline re-run on the cached "
              "extractions (no API calls).", "tab:sens",
              ["Constant", "Value", "$S_{\\mathrm{KG}}$", "$\\Phi_{\\mathrm{prov}}$", "$\\Phi_{\\mathrm{cov}}$", "$\\Phi_{\\mathrm{topo}}$", "Triples", "Entities"],
              sens_rows, "lccccccc"),
        table("Learned weights of the triple-correctness model used by the model-assisted KG-accuracy estimator "
              "(standardised logistic coefficients).", "tab:tripleweights", ["Evidence signal", "Weight"], tw_rows, "lc"),
        table("Individual features: sign-oriented single-feature AUROC on the test split, learned standardised "
              "weight $w_j$ and permutation importance of the champion.", "tab:features",
              ["", "Feature", "Family", "AUROC", "$w_j$", "Perm. imp."], feat_rows, "lllccc"),
        table(r"Per-category behaviour on the test split ($\alpha=\Alpha$).", "tab:percat",
              ["Category", "$n$", "Supported \\%", "mean $S_{\\text{final}}$", "VER", "ABS", "HAL"], cat_rows, "lcccccc"),
        table(r"Extrinsic KG precision per source paper (strict, 95\% Wilson interval; lenient counts partially "
              r"correct triples as correct).", "tab:kgprec",
              ["Paper", "$n$", "Precision", "95\\% CI", "Lenient"], kg_rows, "lcccc"),
    ]
    with open(os.path.join(GEN, "results.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(m) + "\n\n" + "\n".join(tables))
    # placeholders with the SAME macro names, so main.tex compiles even before any run
    import re as _re
    names = list(dict.fromkeys(_re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", "\n".join(m) + "\n".join(tables))))
    with open(os.path.join(GEN, "defaults.tex"), "w", encoding="utf-8") as fh:
        fh.write("% AUTO-GENERATED placeholders: every result macro renders as --\n")
        fh.write("\n".join("\\providecommand{\\" + n + "}{" + ("" if n.startswith("Table") or n.endswith("Sentence") else DASH) + "}" for n in names) + "\n")
    made = figures(pred, prod)  # main.tex guards each figure with \IfFileExists
    z = overleaf_bundle()
    print(f"[paper] wrote {os.path.join(GEN, 'results.tex')} | figures: {made or 'none (no test predictions yet)'} | overleaf: {z}")


if __name__ == "__main__":
    main(sys.argv[1:])
