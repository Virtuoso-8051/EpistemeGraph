r"""
Post-hoc analysis of a finished benchmark run (no API calls, no retraining).

Reads the cached run (answers, judgements, second-judge cross-check, features) and the trained brain, and writes

  <workspace>/benchmark/results_analysis.json   every number quoted in README.md / paper/main.tex
  paper/figures/*.pdf                            publication figures (vector)
  docs/figures/*.png                             the same figures for the GitHub README

Analyses
  1. Label audit      : verdict distribution per category; inter-judge agreement (Cohen's kappa) of the primary
                        and the second judge under every binarisation, and the ordinal (weighted) kappa.
  2. Confounds        : how much answer length alone explains the label (AUROC of length).
  3. Detector vs the stronger judge: AUROC of the out-of-fold detector scores against the SECOND judge's labels on
                        the cross-check sample (does the detector track the more reliable labeller?).
  4. Per-category AUROC of the out-of-fold scores.
  5. Cost accounting  : tokens and estimated USD per model from the run's usage file.

Usage: python scripts/analyze_results.py [--workspace <id>] [--run latest]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve  # noqa: E402

from kg_audit.components.decision import CLAIM_FEATURES, FEATURES, MondrianConformal, empirical_coverage  # noqa: E402
from kg_audit.components.model_trainer import bootstrap_ci, delong_test  # noqa: E402
from kg_audit.utils import workspace as wsm  # noqa: E402
from kg_audit.utils.main_utils import read_json, read_jsonl, write_json_atomic  # noqa: E402

PAPER_FIG = os.path.join(ROOT, "paper", "figures")
DOC_FIG = os.path.join(ROOT, "docs", "figures")
ORDER = {"UNSUPPORTED": 0, "PARTIALLY_SUPPORTED": 1, "SUPPORTED": 2}
COLORS = {"EpistemeGraph_v4": "#2a78d6", "same_learner_v3_inputs": "#7a7a74", "text_nli_checker": "#1baf7a",
          "llm_self_confidence": "#eb6834", "answer_length": "#b8862b"}
LABELS = {"EpistemeGraph_v4": "EpistemeGraph v4", "same_learner_v3_inputs": "same learner, v3 inputs",
          "text_nli_checker": "text-only NLI checker", "llm_self_confidence": "LLM self-confidence", "answer_length": "answer length"}


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def save(fig, name):
    os.makedirs(PAPER_FIG, exist_ok=True)
    os.makedirs(DOC_FIG, exist_ok=True)
    fig.savefig(os.path.join(PAPER_FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(DOC_FIG, name + ".png"), dpi=200, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--run", default="latest")
    args = ap.parse_args(argv)
    W = wsm.workspace_dir(args.workspace or wsm.active_workspace_id())
    B = os.path.join(W, "benchmark")
    R = os.path.join(B, "runs", args.run)
    items = {it["id"]: it for it in read_jsonl(os.path.join(B, "benchmark.jsonl"))}
    judge = {f[:-5]: read_json(os.path.join(R, "judgements", f)) for f in os.listdir(os.path.join(R, "judgements"))}
    xc_dir = next((os.path.join(R, "crosscheck", d) for d in os.listdir(os.path.join(R, "crosscheck"))), None) \
        if os.path.isdir(os.path.join(R, "crosscheck")) else None
    second = {f[:-5]: read_json(os.path.join(xc_dir, f)) for f in os.listdir(xc_dir)} if xc_dir else {}
    feats = pd.DataFrame(read_jsonl(os.path.join(R, "features.jsonl")))
    prod = read_json(os.path.join(W, "production_model_metrics.json"), default={}) or {}
    cvp = pd.read_csv(os.path.join(W, "cv_predictions.csv")) if os.path.exists(os.path.join(W, "cv_predictions.csv")) else None
    out = {"run": R, "n_items": len(items)}

    # ---------------------------------------------------------------- 1. label audit
    v = pd.Series({i: j["verdict"] for i, j in judge.items()})
    cat = pd.Series({i: items[i]["category"] for i in v.index})
    out["verdicts"] = v.value_counts().to_dict()
    out["verdicts_by_category"] = pd.crosstab(cat, v).to_dict(orient="index")
    out["supported_rate_by_category"] = (v == "SUPPORTED").groupby(cat).mean().round(4).to_dict()
    fp = {i: judge[i].get("accepted_false_premise") for i in v.index if items[i]["category"] == "FALSE_PREMISE"}
    out["false_premise_accepted_rate"] = float(np.mean([bool(x) for x in fp.values()])) if fp else None
    if second:
        ids = sorted(set(second) & set(judge))
        a = [judge[i]["verdict"] for i in ids]
        b = [second[i]["verdict"] for i in ids]
        kap = {}
        for name, pos in (("supported_vs_rest", "SUPPORTED"), ("unsupported_vs_rest", "UNSUPPORTED")):
            ya, yb = [int(x == pos) for x in a], [int(x == pos) for x in b]
            kap[name] = {"kappa": float(cohen_kappa_score(ya, yb)), "agreement": float(np.mean(np.array(ya) == np.array(yb)))}
        kap["three_class"] = {"kappa": float(cohen_kappa_score(a, b)), "agreement": float(np.mean(np.array(a) == np.array(b)))}
        kap["ordinal_linear_weighted"] = {"kappa": float(cohen_kappa_score([ORDER[x] for x in a], [ORDER[x] for x in b], weights="linear"))}
        out["judge_agreement"] = {"n": len(ids), "primary": prod.get("judge_model") or "claude-haiku-5-5",
                                  "second": os.path.basename(xc_dir), "kappa": kap,
                                  "confusion_3class[primary_rows x second_cols]": pd.crosstab(pd.Series(a, name="p"), pd.Series(b, name="s")).to_dict()}

    # ---------------------------------------------------------------- 2. confounds
    lab = (v == "SUPPORTED").astype(int)
    lens = pd.Series({i: len(str((read_json(os.path.join(R, "intercepts", f"{i}.json"), default={}) or {}).get("answer", ""))) for i in v.index})
    ncl = feats.set_index("id")["n_claims"].reindex(v.index)
    out["confounds"] = {"answer_length_auroc_for_supported": auc(lab, -lens), "n_claims_auroc_for_supported": auc(lab, -ncl.fillna(0)),
                        "mean_answer_chars_by_verdict": lens.groupby(v).mean().round(1).to_dict()}

    # ---------------------------------------------------------------- 3/4. out-of-fold detector analyses
    if cvp is not None:
        cvp = cvp.set_index("query_id")
        y = cvp["label"].to_numpy()
        models = [c[4:] for c in cvp.columns if c.startswith("oof_")]
        out["oof_auroc"] = {m: auc(y, cvp[f"oof_{m}"]) for m in models}
        out["oof_auroc_by_category"] = {c: {m: auc(g["label"], g[f"oof_{m}"]) for m in models}
                                        for c, g in cvp.groupby("category") if g["label"].nunique() == 2}
        if second:
            ids2 = [i for i in sorted(second) if i in cvp.index]
            y2 = np.array([int(second[i]["verdict"] == "SUPPORTED") for i in ids2])
            y1 = cvp.loc[ids2, "label"].to_numpy()
            out["vs_second_judge_labels"] = {
                "n": len(ids2), "positives": int(y2.sum()),
                "auroc_v4_vs_second_judge": auc(y2, cvp.loc[ids2, "oof_EpistemeGraph_v4"]),
                "auroc_v4_vs_primary_judge_same_items": auc(y1, cvp.loc[ids2, "oof_EpistemeGraph_v4"]),
                "auroc_llm_conf_vs_second_judge": auc(y2, cvp.loc[ids2, "oof_llm_self_confidence"]),
                "auroc_v4_on_items_where_judges_agree": auc(y1[y1 == y2], cvp.loc[np.array(ids2)[y1 == y2], "oof_EpistemeGraph_v4"]),
                "n_agree": int((y1 == y2).sum())}

    # ---------------------------------------------------------------- 4b. risk / abstention trade-off (cross-conformal)
    if cvp is not None:
        from sklearn.model_selection import StratifiedKFold
        q = np.clip(cvp["oof_EpistemeGraph_v4"].to_numpy(float), 0, 1)
        curve = []
        for alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            parts = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=3).split(q, y):
                c = MondrianConformal(alpha=alpha).fit(q[tr], y[tr])
                parts.append((len(te), empirical_coverage(c, q[te], y[te])))
            agg = {k: float(sum(n * r[k] for n, r in parts) / len(y)) for k in
                   ("coverage", "coverage_supported", "coverage_hallucinated", "abstention_rate", "selective_accuracy")}
            curve.append({"alpha": alpha, **agg})
        out["conformal_curve_cv"] = curve

    # ---------------------------------------------------------------- 4b'. "how many match the judge" (out-of-fold, all answers)
    if cvp is not None:
        from sklearn.model_selection import StratifiedKFold
        q = np.clip(cvp["oof_EpistemeGraph_v4"].to_numpy(float), 0, 1)
        pred = (q >= 0.5).astype(int)
        dec = np.array(["ABSTAIN"] * len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=3).split(q, y):   # cross-conformal decisions, alpha=0.10
            c = MondrianConformal(alpha=0.10).fit(q[tr], y[tr])
            dec[te] = [c.decide(float(p)) for p in q[te]]
        committed = dec != "ABSTAIN"
        correct = ((dec == "VERIFIED") & (y == 1)) | ((dec == "HALLUCINATED") & (y == 0))
        tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
        tnr = float((1 - pred[y == 0]).mean()) if (y == 0).any() else float("nan")
        out["matching"] = {
            "n": int(len(y)), "matched_at_0.5": int((pred == y).sum()),
            "majority_baseline": int(max(y.sum(), len(y) - y.sum())),
            "balanced_accuracy": float((tpr + tnr) / 2), "recall_supported": tpr, "recall_hallucinated": tnr,
            "committed": int(committed.sum()), "committed_correct": int(correct.sum()), "abstained": int((~committed).sum()),
            "by_category": {c: {"n": int(g.size), "matched": int((pred[g] == y[g]).sum())}
                            for c in sorted(cvp["category"].unique()) for g in [np.where(cvp["category"].to_numpy() == c)[0]]},
            "note": "out-of-fold: every answer is scored by a model that never saw it",
        }
        res_path = os.path.join(B, "results.csv")
        if os.path.exists(res_path):
            rs = pd.read_csv(res_path)
            m = pd.DataFrame({"id": cvp.index, "oof_S": q.round(4), "oof_decision": dec, "match": (pred == y).astype(int)})
            rs = rs.drop(columns=[c for c in ("oof_S", "oof_decision", "match") if c in rs.columns]).merge(m, on="id", how="left")
            rs.to_csv(res_path, index=False)

    # ---------------------------------------------------------------- 4d. every judge: full-data agreement, third judge, consensus
    arch_dir = os.path.join(R, "judgements_archive")
    others = {}
    if os.path.isdir(arch_dir):
        for d in os.listdir(arch_dir):
            others[d] = {f[:-5]: read_json(os.path.join(arch_dir, d, f)) or {} for f in os.listdir(os.path.join(arch_dir, d))}
    third = read_json(os.path.join(B, "judge_third.json"), default={}) or {}
    if others:
        name, lab = next(iter(others.items()))
        ids = [i for i in v.index if lab.get(i, {}).get("verdict")]
        a = [int(v[i] == "SUPPORTED") for i in ids]
        b = [int(lab[i]["verdict"] == "SUPPORTED") for i in ids]
        out["judge_agreement_full"] = {"other_judge": name, "n": len(ids),
                                       "kappa_binary": float(cohen_kappa_score(a, b)),
                                       "raw_agreement_binary": float(np.mean(np.array(a) == np.array(b))),
                                       "kappa_3class": float(cohen_kappa_score([v[i] for i in ids], [lab[i]["verdict"] for i in ids]))}
        if cvp is not None:
            agree_ids = [i for i, x, z in zip(ids, a, b) if x == z and i in cvp.index]
            ya = cvp.loc[agree_ids, "label"].to_numpy()
            out["consensus_two_judges"] = {"n": len(agree_ids), "positives": int(ya.sum()),
                                           "auroc_v4": auc(ya, cvp.loc[agree_ids, "oof_EpistemeGraph_v4"]),
                                           "auroc_llm": auc(ya, cvp.loc[agree_ids, "oof_llm_self_confidence"]),
                                           "matched_at_0.5": int(((cvp.loc[agree_ids, "oof_EpistemeGraph_v4"] >= 0.5).astype(int) == ya).sum())}
    if third and cvp is not None:
        tv = third.get("verdicts", {})
        ids = [i for i in tv if i in cvp.index]
        yg = np.array([int(tv[i] == "SUPPORTED") for i in ids])
        res3 = {"judge": third.get("third_judge"), "n": len(ids), "supported_rate": float(yg.mean()),
                "auroc_v4_vs_third": auc(yg, cvp.loc[ids, "oof_EpistemeGraph_v4"]),
                "auroc_llm_vs_third": auc(yg, cvp.loc[ids, "oof_llm_self_confidence"]),
                "agreement": third.get("agreement")}
        if others:
            lab = next(iter(others.values()))
            votes = np.array([[int(v[i] == "SUPPORTED"), int(lab.get(i, {}).get("verdict") == "SUPPORTED"), int(tv[i] == "SUPPORTED")] for i in ids])
            ym = (votes.sum(1) >= 2).astype(int)
            unanimous = votes.min(1) == votes.max(1)
            res3.update({"majority_supported_rate": float(ym.mean()),
                         "auroc_v4_vs_majority": auc(ym, cvp.loc[ids, "oof_EpistemeGraph_v4"]),
                         "auroc_llm_vs_majority": auc(ym, cvp.loc[ids, "oof_llm_self_confidence"]),
                         "n_unanimous": int(unanimous.sum()),
                         "auroc_v4_unanimous": auc(ym[unanimous], cvp.loc[np.array(ids)[unanimous], "oof_EpistemeGraph_v4"]),
                         "fleiss_style_all_three_agree": float(unanimous.mean())})
        out["third_judge"] = res3

    # ---------------------------------------------------------------- 4c. label design: binary vs ordinal training target
    lab_path = os.path.join(W, "labelled_features.csv")
    if os.path.exists(lab_path):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.model_selection import RepeatedStratifiedKFold
        from kg_audit.components.decision import FEATURE_KEYS, model_matrix
        L = pd.read_csv(lab_path)
        bmap = {r["query_id"]: r["bag"] for r in read_jsonl(os.path.join(W, "labelled_bags.jsonl"))}
        Xm = model_matrix(L[FEATURE_KEYS].to_numpy(float), [bmap.get(q, []) for q in L["query_id"]])
        yb = L["label"].to_numpy(int)
        yo = L["judge_verdict"].map(ORDER).to_numpy(float)
        ab, ao = [], []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(Xm, yb):
            ab.append(auc(yb[te], ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=2)
                          .fit(Xm[tr], yb[tr]).predict_proba(Xm[te])[:, 1]))
            ao.append(auc(yb[te], ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=2)
                          .fit(Xm[tr], yo[tr]).predict(Xm[te])))
        out["label_design"] = {"protocol": "3x5 repeated stratified CV, ExtraTrees, evaluated on the binary target",
                               "binary_auroc": float(np.mean(ab)), "ordinal_auroc": float(np.mean(ao))}

    # ---------------------------------------------------------------- 5. cost
    usage = read_json(os.path.join(R, "usage.json"), default={}) or {}
    out["cost"] = {"this_run_estimate_usd": usage.get("usd_estimate"), "ledger_total_usd": usage.get("ledger_total_usd"),
                   "per_model": usage.get("per_model")}
    out["kg"] = {k: (read_json(os.path.join(W, "kg_fidelity_state.json"), default={}) or {}).get(k) for k in
                 ("overall_kg_accuracy_score", "verified_triplets_pushed", "resolved_entity_count", "documents_in_corpus",
                  "total_chunks_processed", "cross_pdf_shared_entities_count")}
    write_json_atomic(os.path.join(B, "results_analysis.json"), out)

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(json.dumps(out, indent=2, default=str))
        return
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    made = []
    if cvp is not None:
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        for m in models:
            fpr, tpr, _ = roc_curve(y, cvp[f"oof_{m}"])
            ax.plot(fpr, tpr, lw=1.6 if m == "EpistemeGraph_v4" else 1.0, color=COLORS.get(m),
                    label=f"{LABELS.get(m, m)} ({out['oof_auroc'][m]:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="grey")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(f"Out-of-fold ROC, n={len(y)}", fontsize=9)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        save(fig, "roc_cv"); plt.close(fig); made.append("roc_cv")

        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        bins = np.linspace(0, 1, 9)
        for m in ("EpistemeGraph_v4", "llm_self_confidence"):
            p = np.clip(cvp[f"oof_{m}"].to_numpy(), 0, 1)
            ix = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
            xs = [p[ix == b].mean() for b in range(len(bins) - 1) if (ix == b).sum() >= 5]
            ys = [y[ix == b].mean() for b in range(len(bins) - 1) if (ix == b).sum() >= 5]
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.2, color=COLORS[m], label=LABELS[m])
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="grey")
        ax.set_xlabel("predicted P(supported)"); ax.set_ylabel("observed frequency")
        ax.set_title("Reliability (out-of-fold, bins with n>=5)", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        save(fig, "reliability_cv"); plt.close(fig); made.append("reliability_cv")

    imp, impc = prod.get("permutation_importance") or {}, prod.get("permutation_importance_claims") or {}
    if imp:
        rows = [(f"${ft['tex']}$", imp[ft["key"]]["mean"], imp[ft["key"]]["std"], "#7a7a74") for ft in FEATURES if ft["key"] in imp]
        from export_paper_assets import CLAIM_NAMES
        rows += [("claims: " + CLAIM_NAMES.get(k, k.replace("_", " ")).lower(), v["mean"], v["std"], "#2a78d6") for k, v in impc.items()]
        rows = sorted(rows, key=lambda r: r[1])[-16:]
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.barh(range(len(rows)), [r[1] for r in rows], xerr=[r[2] for r in rows], color=[r[3] for r in rows], ecolor="grey", capsize=2)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.set_xlabel(r"$\Delta$ log-loss when permuted (test split)")
        ax.set_title("Grey: answer features   Blue: pooled claim groups", fontsize=7.5)
        save(fig, "importance_v4"); plt.close(fig); made.append("importance_v4")

    if out.get("conformal_curve_cv"):
        cc = out["conformal_curve_cv"]
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        al = [c["alpha"] for c in cc]
        ax.plot(al, [c["coverage"] for c in cc], marker="o", ms=3.5, color="#2a78d6", label="coverage")
        ax.plot(al, [1 - a for a in al], ls="--", lw=0.8, color="#2a78d6", label=r"target $1-\alpha$")
        ax.plot(al, [c["abstention_rate"] for c in cc], marker="s", ms=3.5, color="#fab219", label="abstention rate")
        ax.plot(al, [c["selective_accuracy"] for c in cc], marker="^", ms=3.5, color="#0ca30c", label="selective accuracy")
        ax.set_xlabel(r"risk level $\alpha$"); ax.set_ylim(0, 1.02)
        ax.set_title("Cross-conformal decisions (n=%d, out-of-fold)" % len(y), fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="center right")
        save(fig, "conformal_tradeoff"); plt.close(fig); made.append("conformal_tradeoff")

    cats = ["FACTUAL", "CROSS_DOC", "BORDERLINE", "FALSE_PREMISE"]
    tab = pd.crosstab(cat, v).reindex(index=cats, columns=["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]).fillna(0)
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    left = np.zeros(len(cats))
    for col, colr in zip(tab.columns, ("#0ca30c", "#fab219", "#d03b3b")):
        ax.barh(range(len(cats)), tab[col], left=left, color=colr, label=col.replace("_", " ").lower())
        left += tab[col].to_numpy()
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([c.replace("_", " ").lower() for c in cats])
    ax.set_xlabel("answers"); ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "labels_by_category"); plt.close(fig); made.append("labels_by_category")
    print(f"[analysis] wrote {os.path.join(B, 'results_analysis.json')} | figures: {made}")


if __name__ == "__main__":
    main(sys.argv[1:])
