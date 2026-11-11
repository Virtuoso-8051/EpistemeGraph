"""
Phase 4 - Self-learning calibration engine (schema casse-v4).

Protocol (every weight and threshold is estimated from labelled data):

  0. Data      : <workspace>/labelled_features.csv - one row per (query, LLM answer) with the 13-D
                 answer vector, S_KG(q) and an ANSWER-level label y in {0,1} (1 = every asserted
                 claim is supported by the papers) + <workspace>/labelled_bags.jsonl - the bag of
                 15-D claim vectors of each answer. No synthetic samples, no self-labelling.
  1. Split     : deterministic hash(query_id) -> train 60% / calibration 20% / test 20%; the test
                 split never changes between retrains (fair champion/challenger comparison).
  2. Tournament: 10 model families (+ optional TabPFN), hyper-parameters tuned by inner 3-fold CV,
                 generalisation estimated by outer 5-fold CV (nested CV, Cawley & Talbot 2010).
                 Selection metric: out-of-fold log-loss (strictly proper scoring rule).
  3. Champion  : tuned on train, then CROSS-FITTED on train+calibration (K-fold): its out-of-fold logits over
                 every non-test answer feed Platt scaling -> S_query; the final model is refit on train+cal.
  4. Pool      : learned log-linear pool of S_query and S_KG(q) -> S_final (fitted on the same out-of-fold set).
  5. Conformal : Mondrian cross-conformal quantiles on those out-of-fold scores (Vovk, 2015): ~5x more
                 calibration points than a split, so the class-conditional quantiles are far more stable.
  5b. CV       : 5x5 repeated stratified CV over ALL labelled answers (champion family, its v3-input twin,
                 text-NLI checker, verbalised confidence, answer length) -> AUROC with spread + DeLong.
  6. Test      : AUROC / log-loss / Brier / ECE with 95% bootstrap CIs, DeLong tests against the
                 LLM's own confidence and against the no-topology ablation, coverage, per-category
                 behaviour, baselines, ablations, permutation importance.
  7. Promotion : the challenger replaces production only if its test log-loss is lower on the SAME split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kg_audit.components.decision import (
    CLAIM_FEATURES,
    FEATURE_KEYS,
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    MODEL_KEYS,
    MODEL_MONOTONE_SIGNS,
    MONOTONE_SIGNS,
    POOLED_KEYS,
    DecisionBundle,
    LogLinearPool,
    MondrianConformal,
    empirical_coverage,
    logit,
    model_matrix,
    sigmoid,
)
from kg_audit.entity.artifact_entity import ModelTrainerArtifact
from kg_audit.entity.config_entity import ModelTrainerConfig
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import limit_cpu_threads, read_json, read_jsonl, write_json_atomic

LABELLED_COLUMNS = (["query_id", "category", "label", "judge_verdict", "s_kg_local"] + FEATURE_KEYS
                    + ["n_claims", "answer_chars", "llm_model", "timestamp", "query"])
# claim features grouped by family (for ablations and grouped importance)
CLAIM_FAMILY = {"sheaf_resid": "topological", "forman_graft": "topological",
                "pol_asserted": "model", "pol_speculative": "model", "pol_refuted": "model"}
TEXT_CLAIM = ["nli_entail", "nli_contra", "nli_cons", "cos_evidence", "ooc_rate"]
GRAPH_CLAIM = ["s_prov", "tier_direct", "tier_two_hop", "conflict", "aligned_frac", "sheaf_resid", "forman_graft"]


def _pooled(claim_keys: List[str]) -> List[str]:
    return [k for k in POOLED_KEYS if k.split("_", 1)[1] in claim_keys]
TELEMETRY_COLUMNS = FEATURE_KEYS + ["s_kg_local", "decision", "s_final", "timestamp", "query"]
D_CLAIM = len(CLAIM_FEATURES)


# ======================================================================================
# Contender: PyTorch feature-attention residual network (answer-level)
# ======================================================================================
class _AttentionResNetModule(nn.Module):
    """Feature-wise softmax gate -> 2 residual MLP blocks -> logit."""

    def __init__(self, in_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_features, in_features), nn.Tanh(),
                                  nn.Linear(in_features, in_features), nn.Softmax(dim=-1))
        self.inp = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        def block():
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        self.b1, self.b2 = block(), block()
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.gate(x)
        h = self.inp(x * (1.0 + a))
        h = self.act(h + self.b1(h))
        h = self.act(h + self.b2(h))
        return self.head(h).squeeze(-1), a


class AttentionResNetClassifier(ClassifierMixin, BaseEstimator):
    """Small-data regime: AdamW + weight decay + early stopping on an internal 15% split."""

    def __init__(self, hidden_dim: int = 32, lr: float = 3e-3, epochs: int = 300,
                 weight_decay: float = 1e-3, dropout: float = 0.1, patience: int = 30, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        idx = rng.permutation(len(y))
        n_val = max(1, int(0.15 * len(y)))
        va, tr = idx[:n_val], idx[n_val:]
        xt, yt = torch.tensor(Xs[tr]), torch.tensor(y[tr], dtype=torch.float32)
        xv, yv = torch.tensor(Xs[va]), torch.tensor(y[va], dtype=torch.float32)
        self.model_ = _AttentionResNetModule(Xs.shape[1], self.hidden_dim, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_w = torch.tensor(max(1e-3, (yt == 0).sum().item()) / max(1e-3, (yt == 1).sum().item()))
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            crit(self.model_(xt)[0], yt).backward()
            opt.step()
            self.model_.eval()
            with torch.no_grad():
                vl = float(nn.functional.binary_cross_entropy_with_logits(self.model_(xv)[0], yv))
            if vl < best - 1e-5:
                best, wait, best_state = vl, 0, {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        with torch.no_grad():
            self.attention_weights_ = self.model_(torch.tensor(Xs))[1].mean(0).numpy()
        return self

    def decision_function(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            return self.model_(torch.tensor(Xs))[0].numpy().astype(np.float64)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ======================================================================================
# Contender: gated-attention multiple-instance network over the CLAIMS of an answer
# ======================================================================================
class _MILModule(nn.Module):
    """
    Instance encoder  h_i = MLP(z_i)                                   (claim i of the answer)
    Gated attention   a_i = softmax_i( w^T [tanh(V h_i) * sigmoid(U h_i)] )   (Ilse et al., 2018)
    Bag embedding     b   = sum_i a_i h_i
    Answer logit      l   = head([ b ; A x ])      (x = 13-D answer-level vector, A linear)
    Claim scores      r_i = sigmoid(c^T h_i)       = P(claim i is unsupported)
    """

    def __init__(self, d_claim: int, d_answer: int, hidden: int, dropout: float):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_claim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.att_v = nn.Linear(hidden, hidden)
        self.att_u = nn.Linear(hidden, hidden)
        self.att_w = nn.Linear(hidden, 1)
        self.inst = nn.Linear(hidden, 1)
        self.ctx = nn.Sequential(nn.Linear(d_answer, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, Z, mask, X):
        h = self.enc(Z)                                                    # (B, N, H)
        s = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1)                                        # (B, N)
        b = (a.unsqueeze(-1) * h).sum(1)                                   # (B, H)
        logit_ = self.head(torch.cat([b, self.ctx(X)], dim=-1)).squeeze(-1)
        r = torch.sigmoid(self.inst(h).squeeze(-1)) * mask                 # claim-level P(unsupported)
        return logit_, a, r


class AttentionMILClassifier(ClassifierMixin, BaseEstimator):
    """
    sklearn-compatible multiple-instance learner. X = [answer features | row index]; the row index points
    into `bag_store` (list of (n_i, 15) claim matrices), so GridSearchCV / cross_val_predict can subset rows.
    Loss = BCE(answer logit, y) + lam * BCE(noisy-OR, y), where the noisy-OR
        P(supported) = prod_i (1 - r_i)
    enforces the multiple-instance assumption "an answer is hallucinated iff some claim is unsupported",
    which is what turns the instance head into a claim-level hallucination detector (weak supervision).
    """

    def __init__(self, bag_store=None, hidden: int = 32, lam: float = 0.5, lr: float = 3e-3, epochs: int = 400,
                 weight_decay: float = 1e-3, dropout: float = 0.1, patience: int = 40, random_state: int = 42):
        self.bag_store = bag_store
        self.hidden = hidden
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state

    # -- tensors
    def _split(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, :-1], X[:, -1].astype(int)

    def _tensors(self, Xa, rows, bags):
        n_max = max([1] + [len(bags[r]) for r in rows])
        Z = np.zeros((len(rows), n_max, D_CLAIM), dtype=np.float32)
        M = np.zeros((len(rows), n_max), dtype=bool)
        for k, r in enumerate(rows):
            b = np.asarray(bags[r], dtype=np.float32).reshape(-1, D_CLAIM) if len(bags[r]) else np.zeros((1, D_CLAIM), np.float32)
            Z[k, :len(b)] = (b - self.z_mu_) / self.z_sd_
            M[k, :len(b)] = True
        Xs = self.x_scaler_.transform(Xa).astype(np.float32)
        return torch.tensor(Z), torch.tensor(M), torch.tensor(Xs)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        Xa, rows = self._split(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        bags = self.bag_store
        allz = np.vstack([np.asarray(bags[r]).reshape(-1, D_CLAIM) for r in rows if len(bags[r])] or [np.zeros((1, D_CLAIM))])
        self.z_mu_, self.z_sd_ = allz.mean(0).astype(np.float32), (allz.std(0) + 1e-6).astype(np.float32)
        self.x_scaler_ = StandardScaler().fit(Xa)
        Z, M, Xs = self._tensors(Xa, rows, bags)
        yt = torch.tensor(y, dtype=torch.float32)
        idx = rng.permutation(len(y))
        n_val = max(1, int(0.15 * len(y)))
        va, tr = torch.tensor(idx[:n_val]), torch.tensor(idx[n_val:])
        self.model_ = _MILModule(D_CLAIM, Xa.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        def loss_on(ix):
            lg, _, r = self.model_(Z[ix], M[ix], Xs[ix])
            p_sup = torch.exp(torch.sum(torch.log(1 - r.clamp(max=1 - 1e-5)) * M[ix], dim=1)).clamp(1e-5, 1 - 1e-5)
            return bce(lg, yt[ix]) + self.lam * nn.functional.binary_cross_entropy(p_sup, yt[ix])

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            loss_on(tr).backward()
            opt.step()
            self.model_.eval()
            with torch.no_grad():
                vl = float(loss_on(va))
            if vl < best - 1e-5:
                best, wait, best_state = vl, 0, {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def _forward(self, X, bags=None):
        Xa, rows = self._split(X)
        Z, M, Xs = self._tensors(Xa, rows, bags if bags is not None else self.bag_store)
        with torch.no_grad():
            lg, a, r = self.model_(Z, M, Xs)
        return lg.numpy().astype(np.float64), a.numpy(), r.numpy(), M.numpy()

    def decision_function(self, X):
        return self._forward(X)[0]

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def claim_outputs(self, Xa: np.ndarray, bags: List[np.ndarray]):
        X = np.column_stack([Xa, np.arange(len(bags))])
        _, a, r, M = self._forward(X, bags)
        return [(a[k][M[k]], r[k][M[k]]) for k in range(len(bags))]


# ======================================================================================
# Production wrapper: one interface for tabular and multiple-instance champions
# ======================================================================================
class ChampionModel(ClassifierMixin, BaseEstimator):
    """
    S_query = sigmoid(a * raw_logit(x) + b), with (a, b) = Platt scaling fitted on the champion's honest
    nested out-of-fold logits. `uses_bags` champions need the claim matrices of the answers.
    """

    def __init__(self, name: str = "", estimator=None, platt_a: float = 1.0, platt_b: float = 0.0, uses_bags: bool = False,
                 localizer=None, pooled: bool = False):
        self.pooled = pooled          # casse-v4: tabular learners see [answer features | pooled claim statistics]
        self.name = name
        self.estimator = estimator
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.uses_bags = uses_bags
        self.localizer = localizer   # multiple-instance network kept for claim-level scores even if it is not champion
        self.classes_ = np.array([0, 1])
        self.eval_bags_: Optional[List[np.ndarray]] = None

    def _raw_logit(self, X, bags=None):
        X = np.asarray(X, dtype=np.float64)
        if self.uses_bags:
            bags = bags if bags is not None else self.eval_bags_
            Xi = np.column_stack([X, np.arange(len(X))])
            return self.estimator._forward(Xi, bags)[0]
        if getattr(self, "pooled", False) and X.shape[1] == len(FEATURE_KEYS):
            X = model_matrix(X, bags if bags is not None else (self.eval_bags_ if self.eval_bags_ is not None else [[]] * len(X)))
        if hasattr(self.estimator, "decision_function") and not hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        return logit(self.estimator.predict_proba(X)[:, 1])

    def predict_proba_matrix(self, Xm):
        """Tabular champions only: score a ready-made [answer | pooled] matrix (grouped importance)."""
        p = sigmoid(self.platt_a * logit(self.estimator.predict_proba(np.asarray(Xm, dtype=np.float64))[:, 1]) + self.platt_b)
        return np.column_stack([1.0 - p, p])

    def predict_proba_bags(self, X, bags):
        p = sigmoid(self.platt_a * self._raw_logit(X, bags) + self.platt_b)
        return np.column_stack([1.0 - p, p])

    def predict_proba(self, X):
        return self.predict_proba_bags(X, None)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def fit(self, X, y):  # interface completeness (permutation_importance never refits)
        return self

    def claim_scores(self, X, bags) -> List[np.ndarray]:
        """P(claim unsupported) for every claim, from the multiple-instance network (champion or companion)."""
        mil = self.estimator if self.uses_bags else self.localizer
        if mil is None:
            return [np.full(len(b), np.nan) for b in bags]
        return [r for _, r in mil.claim_outputs(np.asarray(X, dtype=np.float64), bags)]


# ======================================================================================
# Statistics
# ======================================================================================
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean()) for b in range(n_bins) if (bins == b).any()))


def split_of(query_id: str, test_frac: float, cal_frac: float) -> str:
    h = int(hashlib.sha256(str(query_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "cal" if h < test_frac + cal_frac else "train"


def _safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": _safe_auc(y, p), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p),
            "f1_at_0.5": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
            "accuracy_at_0.5": float(((p >= 0.5).astype(int) == y).mean())}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, B: int = 1000, seed: int = 0) -> Dict[str, List[float]]:
    """95% percentile bootstrap intervals (Efron & Tibshirani, 1993) for the test metrics."""
    rng = np.random.default_rng(seed)
    acc = {"roc_auc": [], "log_loss": [], "brier": [], "ece": []}
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        m = binary_metrics(y[i], p[i])
        for k in acc:
            acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items() if v}


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    t = np.zeros(len(x))
    i = 0
    while i < len(x):
        k = i
        while k < len(x) and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(len(x))
    out[j] = t
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """
    Paired DeLong test (DeLong et al., 1988; fast algorithm of Sun & Xu, 2014) for H0: AUC1 = AUC2.
    Returns both AUCs, their difference, the z statistic and the two-sided p-value.
    """
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return {"auc1": float("nan"), "auc2": float("nan"), "delta": float("nan"), "z": float("nan"), "p_value": float("nan")}
    aucs, v10, v01 = [], [], []
    for p in (np.asarray(p1, float), np.asarray(p2, float)):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    delta = aucs[0] - aucs[1]
    if var <= 1e-15:  # degenerate: identical rankings (or both perfect) -> no evidence of a difference unless delta != 0
        z, p = (0.0, 1.0) if abs(delta) < 1e-12 else (float("inf") * np.sign(delta), 0.0)
    else:
        z = delta / np.sqrt(var)
        p = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": float(delta), "z": float(z), "p_value": p}


# ======================================================================================
# Trainer
# ======================================================================================
class ModelTrainer:
    FEATURE_NAMES = FEATURE_KEYS

    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config = config
            self.production_metrics_path = config.production_metrics_path
            self.n_jobs = max(1, int(config.n_jobs))
            self.bags_path = os.path.join(os.path.dirname(config.labelled_store_path), "labelled_bags.jsonl")
            limit_cpu_threads(self.n_jobs)
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------ data stores
    @staticmethod
    def _append_csv(path: str, columns: List[str], row: Dict[str, Any]) -> int:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != columns:  # schema changed: rotate the legacy file, never mix schemas
                os.replace(path, path.replace(".csv", f"_legacy_{datetime.now():%Y%m%d%H%M%S}.csv"))
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        with open(path, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)

    def append_live_telemetry(self, features: Dict[str, float], s_kg_local: float, decision: str,
                              s_final: Optional[float], query: str) -> int:
        """Unlabelled production traffic: drift monitoring only, NEVER used as training labels."""
        row = {k: round(float(features[k]), 6) for k in FEATURE_KEYS}
        row.update({"s_kg_local": round(float(s_kg_local), 6), "decision": decision,
                    "s_final": "" if s_final is None else round(float(s_final), 6),
                    "timestamp": datetime.now(timezone.utc).isoformat(), "query": query[:300]})
        try:
            return self._append_csv(self.config.telemetry_path, TELEMETRY_COLUMNS, row)
        except Exception as e:
            logging.warning(f"Telemetry logging warning: {e}")
            return 0

    def load_labelled(self) -> Tuple[pd.DataFrame, Dict[str, List[List[float]]]]:
        path = self.config.labelled_store_path
        if not os.path.exists(path):
            return pd.DataFrame(columns=LABELLED_COLUMNS), {}
        df = pd.read_csv(path)
        if not set(FEATURE_KEYS) <= set(df.columns):
            raise ValueError("labelled_features.csv was written with an older feature schema - re-run the benchmark features stage")
        df = df.dropna(subset=FEATURE_KEYS + ["label"]).drop_duplicates(subset=["query_id"], keep="last").reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        df["split"] = [split_of(q, self.config.test_fraction, self.config.calibration_fraction) for q in df["query_id"]]
        bags = {str(r["query_id"]): r.get("bag", []) for r in read_jsonl(self.bags_path)}
        return df, bags

    # ------------------------------------------------------------------ the contenders
    def model_zoo(self, bag_store=None) -> Dict[str, Tuple[Any, Dict[str, list], bool]]:
        """name -> (estimator, grid, uses_bags). Each family covers a distinct inductive bias."""
        rs, sc = self.config.random_state, StandardScaler
        zoo = {
            "LogisticRegression": (Pipeline([("sc", sc()), ("clf", LogisticRegression(max_iter=3000, random_state=rs))]),
                                   {"clf__C": [0.03, 0.1, 0.3, 1.0]}, False),
            "MonotoneHistGBM": (HistGradientBoostingClassifier(monotonic_cst=MODEL_MONOTONE_SIGNS, random_state=rs, early_stopping=False),
                                {"learning_rate": [0.05, 0.1], "max_depth": [2, 3], "max_iter": [150]}, False),
            "RandomForest": (RandomForestClassifier(random_state=rs, n_jobs=1, min_samples_leaf=2),
                             {"n_estimators": [300], "max_depth": [4, 8, None]}, False),
            "ExtraTrees": (ExtraTreesClassifier(random_state=rs, n_jobs=1, min_samples_leaf=2),
                           {"n_estimators": [300], "max_depth": [4, 8, None]}, False),
            "GradientBoosting": (GradientBoostingClassifier(random_state=rs, subsample=0.8),
                                 {"n_estimators": [100, 200], "learning_rate": [0.05], "max_depth": [2, 3]}, False),
            "RBF_SVM": (Pipeline([("sc", sc()), ("svm", SVC(probability=True, random_state=rs))]),
                        {"svm__C": [0.5, 2.0, 8.0], "svm__gamma": ["scale"]}, False),
            "MLP": (Pipeline([("sc", sc()), ("mlp", MLPClassifier(max_iter=2000, early_stopping=True, random_state=rs))]),
                    {"mlp__hidden_layer_sizes": [(32,), (64, 32)], "mlp__alpha": [1e-3, 1e-2]}, False),
            "StackingEnsemble": (StackingClassifier(
                estimators=[("lr", Pipeline([("sc", sc()), ("m", LogisticRegression(max_iter=3000))])),
                            ("hgb", HistGradientBoostingClassifier(monotonic_cst=MODEL_MONOTONE_SIGNS, max_depth=3, random_state=rs)),
                            ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=rs, n_jobs=1)),
                            ("svm", Pipeline([("sc", sc()), ("m", SVC(probability=True, random_state=rs))]))],
                final_estimator=LogisticRegression(max_iter=3000), cv=3, n_jobs=1), {"final_estimator__C": [0.3, 1.0]}, False),
            "AttentionResNet": (AttentionResNetClassifier(random_state=rs), {"hidden_dim": [16, 32], "weight_decay": [1e-3, 1e-2]}, False),
            "AttentionMIL": (AttentionMILClassifier(bag_store=bag_store, random_state=rs),
                             {"hidden": [16, 32], "lam": [0.5, 1.0]}, True),
        }
        if os.getenv("EPISTEME_TABPFN") == "1":  # optional prior-fitted transformer for small tabular data
            try:
                from tabpfn import TabPFNClassifier
                zoo["TabPFN"] = (TabPFNClassifier(device="cpu"), {}, False)
            except Exception as e:
                logging.warning(f"TabPFN requested but unavailable: {e}")
        return zoo

    # ------------------------------------------------------------------ main entry point
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            df, bag_map = self.load_labelled()
            n = len(df)
            counts = df["label"].value_counts().to_dict() if n else {}
            if n < self.config.min_labelled_samples or min(counts.get(0, 0), counts.get(1, 0)) < 10:
                raise ValueError(f"Need >= {self.config.min_labelled_samples} labelled responses with >= 10 per class "
                                 f"(have n={n}, class counts={counts}). Run the benchmark first.")
            bags_all = [np.asarray(bag_map.get(str(q), []), dtype=np.float64).reshape(-1, D_CLAIM) for q in df["query_id"]]
            masks = {s: (df["split"] == s).to_numpy() for s in ("train", "cal", "test")}
            X13 = df[FEATURE_KEYS].to_numpy(float)
            Xall = model_matrix(X13, bags_all)          # [13 answer features | 45 pooled claim statistics]
            yall = df["label"].to_numpy(int)
            idx = {s: np.where(m)[0] for s, m in masks.items()}
            Xtr, ytr = Xall[idx["train"]], yall[idx["train"]]
            Xca, yca = Xall[idx["cal"]], yall[idx["cal"]]
            Xte, yte = Xall[idx["test"]], yall[idx["test"]]
            X13tr, X13te = X13[idx["train"]], X13[idx["test"]]
            btr = [bags_all[i] for i in idx["train"]]
            bca = [bags_all[i] for i in idx["cal"]]
            bte = [bags_all[i] for i in idx["test"]]
            skg = df["s_kg_local"].to_numpy(float)
            print(f"[Phase 4] n={n}: train={len(ytr)} cal={len(yca)} test={len(yte)}", flush=True)

            outer = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            Xtr_mil = np.column_stack([X13tr, np.arange(len(X13tr))])   # MIL pools the claims itself (attention)

            # ---- (2) nested-CV tournament --------------------------------------------------
            leaderboard: Dict[str, Dict[str, Any]] = {}
            oof_store: Dict[str, np.ndarray] = {}
            zoo = self.model_zoo(bag_store=btr)
            for name, (est, grid, uses_bags) in zoo.items():
                print(f"[Phase 4]   - {name} ...", flush=True)
                Xfit = Xtr_mil if uses_bags else Xtr
                try:
                    search = GridSearchCV(est, grid or {}, scoring="neg_log_loss", cv=inner, n_jobs=1 if uses_bags else self.n_jobs,
                                          error_score="raise") if grid else est
                    oof = cross_val_predict(search, Xfit, ytr, cv=outer, method="predict_proba", n_jobs=1)[:, 1]
                    best_params = search.fit(Xfit, ytr).best_params_ if grid else {}
                except Exception as e:
                    logging.warning(f"{name} failed: {e}")
                    print(f"[Phase 4]     {name} skipped: {e}", flush=True)
                    continue
                oof_store[name] = oof
                leaderboard[name] = {"best_params": best_params, "uses_claim_bags": uses_bags, **binary_metrics(ytr, oof)}

            champion = min(leaderboard, key=lambda k: leaderboard[k]["log_loss"])
            est, _, uses_bags = zoo[champion]
            best = leaderboard[champion]["best_params"]

            # ---- (3) cross-fitting on every non-test answer (train + calibration) ----------
            fit_ix = np.concatenate([idx["train"], idx["cal"]])
            Xfit, X13fit, yfit, bfit = Xall[fit_ix], X13[fit_ix], yall[fit_ix], [bags_all[i] for i in fit_ix]
            oof_logit = self._cross_fit_logits(est, best, uses_bags, Xfit, X13fit, yfit, bfit)
            final_est = clone(est).set_params(**best)
            if uses_bags:
                final_est.set_params(bag_store=bfit)
                final_est.fit(np.column_stack([X13fit, np.arange(len(X13fit))]), yfit)
            else:
                final_est.fit(Xfit, yfit)
            platt = LogisticRegression(C=1e6, max_iter=2000).fit(oof_logit.reshape(-1, 1), yfit)
            localizer = None
            if not uses_bags and "AttentionMIL" in leaderboard:   # companion claim localiser
                localizer = clone(zoo["AttentionMIL"][0]).set_params(**leaderboard["AttentionMIL"]["best_params"], bag_store=bfit)
                localizer.fit(np.column_stack([X13fit, np.arange(len(X13fit))]), yfit)
            model = ChampionModel(champion, final_est, float(platt.coef_[0, 0]), float(platt.intercept_[0]), uses_bags, localizer,
                                  pooled=not uses_bags)
            s_q_oof = sigmoid(model.platt_a * oof_logit + model.platt_b)

            # ---- (4) learned log-linear pool on the same out-of-fold S_query -------------------
            pool = LogLinearPool().fit(s_q_oof, skg[fit_ix], yfit)

            # ---- (5) Mondrian CROSS-conformal on the out-of-fold scores ------------------------
            conformal = MondrianConformal(alpha=self.config.conformal_alpha).fit(pool(s_q_oof, skg[fit_ix]), yfit)
            bundle = DecisionBundle(pool=pool, conformal=conformal)

            # ---- (6) held-out test evaluation ------------------------------------------------
            s_q_te = model.predict_proba_bags(X13te, bte)[:, 1]
            s_f_te = pool(s_q_te, skg[idx["test"]])
            c_llm_te = X13te[:, FEATURE_KEYS.index("c_llm")]
            abl_preds, ablations = self._ablations(Xall[fit_ix], yfit, Xte, yte)
            test = {
                "S_final": binary_metrics(yte, s_f_te),
                "S_final_ci95": bootstrap_ci(yte, s_f_te),
                "S_query": binary_metrics(yte, s_q_te),
                "conformal": empirical_coverage(conformal, s_f_te, yte),
                "baselines": self._baselines(X13te, yte, df.iloc[idx["test"]]),
                "ablations": ablations,
                "significance": {
                    "vs_llm_self_confidence": delong_test(yte, s_f_te, c_llm_te),
                    "vs_text_nli_only": delong_test(yte, s_f_te, abl_preds["text_nli_only"]),
                    "all_vs_without_topological": delong_test(yte, abl_preds["all_features"], abl_preds["without_topological"]),
                },
                "per_category": self._per_category(df.iloc[idx["test"]], s_f_te, conformal),
                "roc_curve": self._roc_points(yte, s_f_te),
                "n_test": int(len(yte)),
            }
            weights = self._interpretable_weights(X13[fit_ix], yfit)
            importance, importance_claims = self._grouped_importance(model, X13te, bte, yte)
            print("[Phase 4] 5x5 repeated cross-validation over all labelled answers ...", flush=True)
            cv_metrics = self._repeated_cv(df, Xall, X13, yall, bags_all, est, best, uses_bags, out_dir=os.path.dirname(self.config.trained_model_file_path))

            # ---- (7) promotion --------------------------------------------------------------
            prod_ll = self._production_test_logloss(X13te, bte, yte, skg[idx["test"]])
            challenger_ll = test["S_final"]["log_loss"]
            promoted = prod_ll is None or challenger_ll < prod_ll

            payload = {
                "schema_version": FEATURE_SCHEMA_VERSION, "feature_schema": FEATURES, "claim_features": CLAIM_FEATURES,
                "winning_model": champion, "winning_params": leaderboard[champion]["best_params"],
                "selection_metric": "nested-CV out-of-fold log-loss (lower is better)",
                "platt": {"a": model.platt_a, "b": model.platt_b},
                "n_labelled": n, "class_counts": {str(k): int(v) for k, v in counts.items()},
                "split_sizes": {k: int(len(v)) for k, v in idx.items()},
                "leaderboard": leaderboard, "decision_bundle": bundle.to_dict(),
                "learned_feature_weights": weights, "permutation_importance": importance,
                "permutation_importance_claims": importance_claims, "model_inputs": MODEL_KEYS,
                "calibration": "cross-fitted on train+cal (K-fold out-of-fold) - Platt, pool and Mondrian cross-conformal",
                "test_metrics": test, "cv_metrics": cv_metrics, "challenger_test_log_loss": challenger_ll,
                "production_test_log_loss_before": prod_ll, "is_promoted_to_production": promoted,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            out_dir = os.path.dirname(self.config.trained_model_file_path)
            os.makedirs(out_dir, exist_ok=True)
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)
            write_json_atomic(self.config.metrics_report_file_path, payload, indent=4)
            te = df.iloc[idx["test"]]
            pred = pd.DataFrame({"query_id": te["query_id"].values, "category": te["category"].values, "label": yte,
                                 "s_query": s_q_te, "s_final": s_f_te, "c_llm": c_llm_te,
                                 "decision": [conformal.decide(p) for p in s_f_te]})
            pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

            if promoted:
                with open(self.config.final_model_file_path, "wb") as f:
                    pickle.dump(model, f)
                write_json_atomic(self.production_metrics_path, payload, indent=4)
                pred.to_csv(os.path.join(os.path.dirname(self.config.final_model_file_path), "test_predictions.csv"), index=False)
                cvp = os.path.join(out_dir, "cv_predictions.csv")
                if os.path.exists(cvp):
                    import shutil
                    shutil.copy2(cvp, os.path.join(os.path.dirname(self.config.final_model_file_path), "cv_predictions.csv"))
                print(f"[Phase 4] PROMOTED {champion}: test log-loss {challenger_ll:.4f} (prod was {prod_ll})", flush=True)
            else:
                print(f"[Phase 4] Kept production brain: challenger {challenger_ll:.4f} >= production {prod_ll:.4f}", flush=True)

            m = test["S_final"]
            return ModelTrainerArtifact(
                best_model_name=champion, trained_model_file_path=self.config.trained_model_file_path,
                roc_auc_score=m["roc_auc"], f1_score=m["f1_at_0.5"], brier_score=m["brier"],
                expected_calibration_error=m["ece"], log_loss=m["log_loss"], is_model_accepted=promoted,
                n_labelled_samples=n, conformal=conformal.to_dict(), pool=pool.to_dict(),
                learned_feature_weights=weights["standardised_coefficients"], test_metrics=test)
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------ cross-fitting / repeated CV
    def _fit_predict_logit(self, est, params, uses_bags, Xm_tr, X13_tr, y_tr, b_tr, Xm_te, X13_te, b_te) -> np.ndarray:
        e = clone(est).set_params(**params)
        if uses_bags:
            e.set_params(bag_store=b_tr)
            e.fit(np.column_stack([X13_tr, np.arange(len(X13_tr))]), y_tr)
            return e._forward(np.column_stack([X13_te, np.arange(len(X13_te))]), b_te)[0]
        e.fit(Xm_tr, y_tr)
        return logit(e.predict_proba(Xm_te)[:, 1])

    def _cross_fit_logits(self, est, params, uses_bags, Xm, X13m, y, bags) -> np.ndarray:
        kf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        out = np.zeros(len(y))
        for tr, te in kf.split(Xm, y):
            out[te] = self._fit_predict_logit(est, params, uses_bags, Xm[tr], X13m[tr], y[tr], [bags[i] for i in tr],
                                              Xm[te], X13m[te], [bags[i] for i in te])
        return out

    def _repeated_cv(self, df, Xall, X13, y, bags, est, params, uses_bags, out_dir: str, n_rep: int = 5) -> Dict[str, Any]:
        """
        5 x 5 repeated stratified CV over every labelled answer: far tighter than one 20% test split.
        Contenders: the champion family (v4 inputs), the SAME family on the v3 answer features only,
        a text-only NLI checker, the LLM's verbalised confidence and an answer-length baseline.
        Also a secondary target: severe hallucination (verdict UNSUPPORTED) vs the rest.
        """
        from sklearn.base import clone as _clone
        rs = self.config.random_state
        lr = lambda C=0.3: Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=3000))])  # noqa: E731
        col = lambda keys: [MODEL_KEYS.index(k) for k in keys]  # noqa: E731
        text_cols = col(["sigma_ent", "nli_cons", "nli_worst", "nu_ooc"] + _pooled(TEXT_CLAIM))
        has_len = {"n_claims", "answer_chars"} <= set(df.columns) and df[["n_claims", "answer_chars"]].notna().all().all()
        L = np.column_stack([np.log1p(df["answer_chars"].to_numpy(float)), df["n_claims"].to_numpy(float)]) if has_len else None
        c_llm = X13[:, FEATURE_KEYS.index("c_llm")]
        rcv = RepeatedStratifiedKFold(n_splits=self.config.cv_folds, n_repeats=n_rep, random_state=rs)
        names = ["EpistemeGraph_v4", "same_learner_v3_inputs", "text_nli_checker", "llm_self_confidence"] + (["answer_length"] if has_len else [])
        oof = {k: np.zeros((n_rep, len(y))) for k in names}
        fold_auc = {k: [] for k in names}
        tab_est = est if not uses_bags else Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=3000))])
        tab_params = params if not uses_bags else {}
        for f, (tr, te) in enumerate(rcv.split(Xall, y)):
            r = f // self.config.cv_folds
            pr = {"EpistemeGraph_v4": sigmoid(self._fit_predict_logit(est, params, uses_bags, Xall[tr], X13[tr], y[tr], [bags[i] for i in tr],
                                                                        Xall[te], X13[te], [bags[i] for i in te])),
                  "same_learner_v3_inputs": self._v3_twin(tab_est, tab_params).fit(X13[tr], y[tr]).predict_proba(X13[te])[:, 1]
                  if not uses_bags else lr().fit(X13[tr], y[tr]).predict_proba(X13[te])[:, 1],
                  "text_nli_checker": lr().fit(Xall[tr][:, text_cols], y[tr]).predict_proba(Xall[te][:, text_cols])[:, 1],
                  "llm_self_confidence": c_llm[te]}
            if has_len:
                pr["answer_length"] = lr(1.0).fit(L[tr], y[tr]).predict_proba(L[te])[:, 1]
            for k, v in pr.items():
                oof[k][r, te] = v
                fold_auc[k].append(_safe_auc(y[te], v))
        res: Dict[str, Any] = {"protocol": f"{n_rep}x{self.config.cv_folds} repeated stratified CV, n={len(y)}", "n": int(len(y)),
                               "positives": int(y.sum()), "models": {}}
        for k in names:
            rep_auc = [_safe_auc(y, oof[k][r]) for r in range(n_rep)]
            m0 = binary_metrics(y, np.clip(oof[k][0], 0, 1))
            res["models"][k] = {"auroc_mean": float(np.nanmean(fold_auc[k])), "auroc_sd_folds": float(np.nanstd(fold_auc[k])),
                                "auroc_pooled_oof": float(np.mean(rep_auc)), "auroc_ci95_rep0": bootstrap_ci(y, np.clip(oof[k][0], 0, 1), B=1000)["roc_auc"],
                                "log_loss_rep0": m0["log_loss"], "brier_rep0": m0["brier"], "ece_rep0": m0["ece"]}
        res["delong_rep0"] = {f"v4_vs_{k}": delong_test(y, oof["EpistemeGraph_v4"][0], oof[k][0]) for k in names if k != "EpistemeGraph_v4"}
        # secondary target: severe hallucination (the core of the answer is unsupported)
        if "judge_verdict" in df.columns:
            y2 = (df["judge_verdict"].astype(str).str.upper() != "UNSUPPORTED").astype(int).to_numpy()
            if 10 <= y2.sum() <= len(y2) - 10:
                o2 = np.zeros(len(y2))
                for tr, te in StratifiedKFold(self.config.cv_folds, shuffle=True, random_state=rs).split(Xall, y2):
                    o2[te] = sigmoid(self._fit_predict_logit(est, params, uses_bags, Xall[tr], X13[tr], y2[tr], [bags[i] for i in tr],
                                                             Xall[te], X13[te], [bags[i] for i in te]))
                res["severe_hallucination_target"] = {"definition": "y=1 unless verdict is UNSUPPORTED", "positives": int(y2.sum()),
                                                      "auroc": _safe_auc(y2, o2), "auroc_ci95": bootstrap_ci(y2, o2, B=1000)["roc_auc"],
                                                      "llm_self_confidence_auroc": _safe_auc(y2, c_llm)}
        try:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"query_id": df["query_id"].values, "category": df["category"].values, "label": y,
                          "judge_verdict": df.get("judge_verdict", pd.Series([""] * len(y))).values,
                          **{f"oof_{k}": oof[k][0] for k in names}}).to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)
        except Exception as e:
            logging.warning(f"cv_predictions.csv not written: {e}")
        return res

    @staticmethod
    def _v3_twin(est, params):
        """The same learner and hyper-parameters, re-targeted to the 13 v3 answer features."""
        e = clone(est).set_params(**params)
        mono = {k: MONOTONE_SIGNS for k in e.get_params(deep=True) if k.endswith("monotonic_cst")}
        return e.set_params(**mono) if mono else e

    def _grouped_importance(self, model, X13, bags, y, repeats: int = 20):
        """Permutation importance (log-loss increase) of the 13 answer features and of each claim feature's pooled group."""
        rng = np.random.default_rng(self.config.random_state)
        ll = lambda p: float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))  # noqa: E731
        base = ll(model.predict_proba_bags(X13, bags)[:, 1])
        imp13, impc = {}, {}
        for j, k in enumerate(FEATURE_KEYS):
            d = []
            for _ in range(repeats):
                Xp = X13.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                d.append(ll(model.predict_proba_bags(Xp, bags)[:, 1]) - base)
            imp13[k] = {"mean": float(np.mean(d)), "std": float(np.std(d))}
        if getattr(model, "pooled", False) and not model.uses_bags:
            Xm = model_matrix(X13, bags)
            for c in CLAIM_FEATURES:
                cols = [MODEL_KEYS.index(f"{st}_{c}") for st in ("mean", "min", "max")]
                d = []
                for _ in range(repeats):
                    Xp = Xm.copy(); perm = rng.permutation(len(Xp)); Xp[:, cols] = Xp[perm][:, cols]
                    d.append(ll(model.predict_proba_matrix(Xp)[:, 1]) - base)
                impc[c] = {"mean": float(np.mean(d)), "std": float(np.std(d))}
        return imp13, impc

    # ------------------------------------------------------------------ analysis helpers
    def _interpretable_weights(self, X, y) -> Dict[str, Any]:
        """The learned multipliers: logit P(y=1|x) = b + sum_j w_j (x_j - mu_j)/sd_j."""
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=3000))]).fit(X, y)
        coef = pipe.named_steps["clf"].coef_[0]
        return {"standardised_coefficients": dict(zip(FEATURE_KEYS, map(float, coef))),
                "intercept": float(pipe.named_steps["clf"].intercept_[0]),
                "feature_means": dict(zip(FEATURE_KEYS, map(float, pipe.named_steps["sc"].mean_))),
                "feature_scales": dict(zip(FEATURE_KEYS, map(float, pipe.named_steps["sc"].scale_)))}

    @staticmethod
    def _baselines(Xte, yte, te_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        out = {}
        for j, f in enumerate(FEATURES):
            sign = f["sign"] if f["sign"] != 0 else 1
            out[f"single::{f['key']}"] = {"roc_auc": _safe_auc(yte, sign * Xte[:, j])}
        c = np.clip(Xte[:, FEATURE_KEYS.index("c_llm")], 0, 1)
        out["llm_self_confidence"] = {**binary_metrics(yte, c), "ci95": bootstrap_ci(yte, c)}
        if te_df is not None and "answer_chars" in te_df.columns and te_df["answer_chars"].notna().all():
            out["single::answer_length"] = {"roc_auc": _safe_auc(yte, -te_df["answer_chars"].to_numpy(float))}
        return out

    @staticmethod
    def _ablations(Xtr, ytr, Xte, yte) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Same learner (standardised L2 logistic regression) on input subsets: what does each family contribute?"""
        fam = lambda f: [x["key"] for x in FEATURES if x["family"] == f]  # noqa: E731
        topo = fam("topological") + _pooled(["sheaf_resid", "forman_graft"])
        modl = fam("model") + _pooled(["pol_asserted", "pol_speculative", "pol_refuted"])
        groups = {
            "all_features": MODEL_KEYS,
            "answer_features_only_v3": FEATURE_KEYS,
            "pooled_claims_only": POOLED_KEYS,
            "topological_only": topo,
            "evidential_only": fam("evidential") + _pooled([k for k in CLAIM_FEATURES if k not in CLAIM_FAMILY]),
            "model_signals_only": modl,
            "text_nli_only": ["sigma_ent", "nli_cons", "nli_worst", "nu_ooc"] + _pooled(TEXT_CLAIM),   # literature-style checker
            "graph_only": fam("topological") + ["pi_prov", "kappa_con", "align_rate"] + _pooled(GRAPH_CLAIM),
            "without_topological": [k for k in MODEL_KEYS if k not in topo],
            "without_model_signals": [k for k in MODEL_KEYS if k not in modl],
        }
        preds, out = {}, {}
        for g, keys in groups.items():
            ix = [MODEL_KEYS.index(k) for k in keys]
            m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=3000))]).fit(Xtr[:, ix], ytr)
            p = m.predict_proba(Xte[:, ix])[:, 1]
            preds[g] = p
            out[g] = {**binary_metrics(yte, p), "ci95": bootstrap_ci(yte, p, B=500), "features": keys}
        return preds, out

    @staticmethod
    def _per_category(te: pd.DataFrame, s_final: np.ndarray, conformal: MondrianConformal) -> Dict[str, Any]:
        out = {}
        for cat in sorted(te["category"].astype(str).unique()):
            m = (te["category"].astype(str) == cat).to_numpy()
            dec = [conformal.decide(p) for p in s_final[m]]
            out[cat] = {"n": int(m.sum()), "label_rate": float(te["label"].to_numpy()[m].mean()),
                        "mean_S_final": float(np.mean(s_final[m])),
                        "decisions": {d: int(dec.count(d)) for d in ("VERIFIED", "ABSTAIN", "HALLUCINATED")}}
        return out

    @staticmethod
    def _roc_points(y, p, max_points: int = 60) -> Dict[str, List[float]]:
        if len(np.unique(y)) < 2:
            return {"fpr": [], "tpr": []}
        fpr, tpr, _ = roc_curve(y, p)
        i = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
        return {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist()}

    def _production_test_logloss(self, Xte, bte, yte, skg) -> Optional[float]:
        prod = read_json(self.production_metrics_path, default=None)
        if not prod or prod.get("schema_version") != FEATURE_SCHEMA_VERSION or not os.path.exists(self.config.final_model_file_path):
            return None
        try:
            with open(self.config.final_model_file_path, "rb") as f:
                model = pickle.load(f)
            bundle = DecisionBundle.from_dict(prod.get("decision_bundle"))
            sq = model.predict_proba_bags(Xte, bte)[:, 1] if hasattr(model, "predict_proba_bags") else model.predict_proba(Xte)[:, 1]
            return float(log_loss(yte, np.clip(bundle.pool(sq, skg), 1e-6, 1 - 1e-6), labels=[0, 1]))
        except Exception as e:
            logging.warning(f"Could not score production model: {e}")
            return None
