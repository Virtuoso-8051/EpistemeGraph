"""
Decision layer of EpistemeGraph (Phase 4 -> Phase 3 hand-off).

Replaces every hard-coded constant of the previous version:

    OLD   S_final = 0.2 * S_KG + 0.8 * S_query             (fixed weights)
          VERIFIED if S_final > 0.85, HALLUCINATED if <= 0.5 (fixed thresholds)

    NEW   (1) S_query  = calibrated P(y=1 | x),  x in R^58  (13 answer features + 45 pooled claim statistics)
          (2) S_final  = learned log-linear opinion pool of S_query and the *local*
                         KG fidelity S_KG(q) of the retrieved subgraph:
                    logit S_final = b0 + b1 * logit S_query + b2 * logit S_KG(q)
              The fixed geometric pool S_query^eta * S_KG^(1-eta) is the special case
              b0 = 0, b1 = eta, b2 = 1 - eta, so the old formula is *nested* in the new one.
          (3) Decision = Mondrian (class-conditional) split-conformal prediction set
                    Gamma(x) = { c in {0,1} : 1 - p_c(x) <= q_c }
              with q_c the ceil((n_c+1)(1-alpha))-th smallest out-of-fold non-conformity score of
              class c.  |Gamma| = 1 and its class is the more probable one -> commit, otherwise ABSTAIN.
              Guarantee (exchangeability): P(y in Gamma(x) | y = c) >= 1 - alpha for each c.

The only user-facing knob is the risk level alpha (e.g. 0.10), which is a statistical
guarantee, not a heuristic score threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------------------
# Feature schema (single source of truth for evaluator, trainer, UI, paper)
# ----------------------------------------------------------------------------------------
FEATURE_SCHEMA_VERSION = "casse-v4"

# Answer-level vector x in R^13. (key, names, LaTeX, expected monotone direction w.r.t. P(supported), family)
#   topological : how the claim graph sits in the geometry of the knowledge graph
#   evidential  : what the source papers say about the claims (graph provenance, raw text, NLI)
#   model       : what the LLM itself reveals (verbalised confidence, token uncertainty, hedging)
FEATURES: List[Dict[str, object]] = [
    {"key": "e_sheaf", "short": "Sheaf energy", "name": "Sheaf Laplacian energy", "tex": r"\mathcal{E}_{\mathcal{F}}", "sign": -1, "family": "topological"},
    {"key": "delta_k", "short": "Ricci curvature shift", "name": "Forman-Ricci curvature shift", "tex": r"\Delta\kappa", "sign": -1, "family": "topological"},
    {"key": "d_spec", "short": "Spectral divergence", "name": "Spectral Wasserstein divergence", "tex": r"D_{\mathrm{spec}}", "sign": -1, "family": "topological"},
    {"key": "pi_prov", "short": "Provenance support", "name": "Bayesian provenance support", "tex": r"\pi_{\mathrm{prov}}", "sign": +1, "family": "evidential"},
    {"key": "sigma_ent", "short": "Evidence similarity", "name": "Evidence embedding similarity", "tex": r"\sigma_{\mathrm{sim}}", "sign": +1, "family": "evidential"},
    {"key": "nli_cons", "short": "NLI consistency", "name": "Mean NLI consistency (SummaC-ZS)", "tex": r"\bar\gamma_{\mathrm{NLI}}", "sign": +1, "family": "evidential"},
    {"key": "nli_worst", "short": "Worst-claim NLI", "name": "Worst-claim NLI consistency", "tex": r"\gamma^{\min}_{\mathrm{NLI}}", "sign": +1, "family": "evidential"},
    {"key": "nu_ooc", "short": "Out-of-corpus novelty", "name": "Out-of-corpus novelty (IDF)", "tex": r"\nu_{\mathrm{ooc}}", "sign": -1, "family": "evidential"},
    {"key": "kappa_con", "short": "Contradiction rate", "name": "Graph contradiction rate", "tex": r"\kappa_{\mathrm{con}}", "sign": -1, "family": "evidential"},
    {"key": "align_rate", "short": "Entity alignment", "name": "Entity alignment rate", "tex": r"\rho_{\mathrm{align}}", "sign": +1, "family": "evidential"},
    {"key": "c_llm", "short": "LLM self-confidence", "name": "LLM verbalised confidence", "tex": r"c_{\mathrm{LLM}}", "sign": 0, "family": "model"},
    {"key": "u_nll", "short": "Token uncertainty", "name": "Mean token negative log-likelihood", "tex": r"\bar u_{\mathrm{NLL}}", "sign": 0, "family": "model"},
    {"key": "spec_ratio", "short": "Hedging ratio", "name": "Speculative-claim ratio", "tex": r"h_{\mathrm{spec}}", "sign": 0, "family": "model"},
]
FEATURE_KEYS: List[str] = [f["key"] for f in FEATURES]
MONOTONE_SIGNS: List[int] = [int(f["sign"]) for f in FEATURES]

# Claim-level vector z_i in R^15 for every claim of an answer. An answer is a BAG of claims; it is
# hallucinated iff at least one claim is unsupported (the standard multiple-instance assumption), so a
# multiple-instance network can learn claim-level hallucination scores from answer-level labels only.
CLAIM_FEATURES: List[str] = [
    "s_prov", "tier_direct", "tier_two_hop", "conflict", "pol_asserted", "pol_speculative", "pol_refuted",
    "aligned_frac", "nli_entail", "nli_contra", "nli_cons", "cos_evidence", "ooc_rate", "sheaf_resid", "forman_graft",
]

# casse-v4: permutation-invariant summary of the claim bag (Deep Sets with fixed pooling, Zaheer et al. 2017).
# For every claim feature we take mean / min / max over the answer's claims: "min nli_entail" is the weakest
# claim's entailment, "max ooc_rate" the most out-of-corpus claim, "mean tier_direct" the share of claims with a
# direct KG edge, and so on. The answer-level learners see x = [13 answer features | 45 pooled claim statistics].
POOL_STATS = ("mean", "min", "max")
POOLED_KEYS: List[str] = [f"{s}_{k}" for k in CLAIM_FEATURES for s in POOL_STATS]
MODEL_KEYS: List[str] = FEATURE_KEYS + POOLED_KEYS
MODEL_MONOTONE_SIGNS: List[int] = MONOTONE_SIGNS + [0] * len(POOLED_KEYS)


def pool_claims(bag) -> np.ndarray:
    """45-D pooled statistics of one claim bag (zeros for an answer without extractable claims)."""
    b = np.asarray(bag if bag is not None else [], dtype=np.float64).reshape(-1, len(CLAIM_FEATURES))
    if not len(b):
        return np.zeros(len(POOLED_KEYS))
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([b.mean(0), b.min(0), b.max(0)]).reshape(-1)


def model_matrix(X_answer, bags) -> np.ndarray:
    """[13 answer features | 45 pooled claim statistics] for every answer."""
    X_answer = np.asarray(X_answer, dtype=np.float64).reshape(-1, len(FEATURE_KEYS))
    bags = list(bags) if bags is not None else [[]] * len(X_answer)
    return np.column_stack([X_answer, np.vstack([pool_claims(b) for b in bags]) if len(bags) else np.zeros((0, len(POOLED_KEYS)))])


DECISIONS = ("VERIFIED", "ABSTAIN", "HALLUCINATED")
UNCALIBRATED = "UNCALIBRATED"

_EPS = 1e-6


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


# ----------------------------------------------------------------------------------------
# (2) Learned log-linear opinion pool  S_final = pool(S_query, S_KG(q))
# ----------------------------------------------------------------------------------------
@dataclass
class LogLinearPool:
    b0: float = 0.0
    b_query: float = 1.0
    b_kg: float = 0.0
    fitted: bool = False
    n_fit: int = 0
    kg_term_used: bool = False      # kept only if (i) the LRT rejects b_kg = 0 AND (ii) it lowers cross-validated log-loss
    lrt_chi2: float = 0.0
    lrt_p: float = 1.0
    cv_logloss_with_kg: float = float("nan")
    cv_logloss_without_kg: float = float("nan")

    def fit(self, s_query: Sequence[float], s_kg: Sequence[float], y: Sequence[int], l2: float = 1e-2) -> "LogLinearPool":
        """
        Maximum-likelihood fit of a 3-parameter logistic model on logits, with a weak
        Gaussian prior (L2) for stability. s_query MUST be out-of-fold predictions,
        otherwise the pool learns the champion's in-sample over-confidence.
        Solved by Newton-Raphson (IRLS) - converges in < 25 iterations, no extra deps.
        """
        y = np.asarray(y, dtype=np.float64)
        Z3 = np.column_stack([np.ones_like(y), logit(s_query), logit(s_kg)])
        if len(np.unique(y)) < 2 or len(y) < 10:
            return self

        def irls(Z, yy):
            w = np.zeros(Z.shape[1])
            w[1] = 1.0
            reg = l2 * np.eye(Z.shape[1])
            reg[0, 0] = 0.0  # never shrink the intercept
            for _ in range(50):
                p = sigmoid(Z @ w)
                W = p * (1.0 - p) + 1e-9
                step = np.linalg.solve((Z * W[:, None]).T @ Z + reg, Z.T @ (p - yy) + reg @ w)
                w -= step
                if np.max(np.abs(step)) < 1e-8:
                    break
            return w

        def nll(Z, w, yy):
            p = np.clip(sigmoid(Z @ w), 1e-12, 1 - 1e-12)
            return float(-np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

        w2 = irls(Z3[:, :2], y)
        self.b0, self.b_query, self.b_kg, self.kg_term_used = float(w2[0]), float(w2[1]), 0.0, False
        # S_KG(q) may be constant (one tiny KG): then b_kg is unidentifiable (Prop. pool b) and is never added.
        if np.std(Z3[:, 2]) > 1e-6:
            w3 = irls(Z3, y)
            self.lrt_chi2 = max(0.0, 2.0 * (nll(Z3[:, :2], w2, y) - nll(Z3, w3, y)))
            self.lrt_p = float(math.erfc(math.sqrt(self.lrt_chi2 / 2.0)))     # chi-square(1) survival function
            # out-of-sample check with the same criterion as model selection: 5 x 5-fold CV log-loss
            n, l_with, l_without = len(y), 0.0, 0.0
            for rep in range(5):
                rng = np.random.default_rng(rep)
                fold = np.empty(n, dtype=int)
                for cls in (0, 1):                               # stratified fold assignment
                    ix = rng.permutation(np.where(y == cls)[0])
                    fold[ix] = np.arange(len(ix)) % 5
                for k in range(5):
                    tr, te = fold != k, fold == k
                    l_with += nll(Z3[te], irls(Z3[tr], y[tr]), y[te])
                    l_without += nll(Z3[te][:, :2], irls(Z3[tr][:, :2], y[tr]), y[te])
            self.cv_logloss_with_kg, self.cv_logloss_without_kg = l_with / (5 * n), l_without / (5 * n)
            if self.lrt_p < 0.05 and self.cv_logloss_with_kg < self.cv_logloss_without_kg:   # the KG term earns its place
                self.b0, self.b_query, self.b_kg, self.kg_term_used = float(w3[0]), float(w3[1]), float(w3[2]), True
        self.fitted, self.n_fit = True, int(len(y))
        return self

    def __call__(self, s_query: float | np.ndarray, s_kg: float | np.ndarray) -> np.ndarray:
        z = self.b0 + self.b_query * logit(s_query) + self.b_kg * logit(s_kg)
        return sigmoid(z)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "LogLinearPool":
        keys = {"b0", "b_query", "b_kg", "fitted", "n_fit", "kg_term_used", "lrt_chi2", "lrt_p",
                "cv_logloss_with_kg", "cv_logloss_without_kg"}
        return cls(**{k: v for k, v in d.items() if k in keys}) if d else cls()


# ----------------------------------------------------------------------------------------
# (3) Mondrian split-conformal tri-state decision
# ----------------------------------------------------------------------------------------
def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0  # no information -> always include the class (forces ABSTAIN)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


@dataclass
class MondrianConformal:
    alpha: float = 0.10
    q_hall: float = 1.0   # q_0 : quantile of non-conformity scores for y = 0 (hallucinated)
    q_ver: float = 1.0    # q_1 : quantile of non-conformity scores for y = 1 (supported)
    n_cal_hall: int = 0
    n_cal_ver: int = 0
    fitted: bool = False

    def fit(self, p_supported: Sequence[float], y: Sequence[int]) -> "MondrianConformal":
        p = np.asarray(p_supported, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        s0 = p[y == 0]            # non-conformity of class 0 = 1 - p_0 = p
        s1 = 1.0 - p[y == 1]      # non-conformity of class 1 = 1 - p_1
        self.q_hall = _conformal_quantile(s0, self.alpha)
        self.q_ver = _conformal_quantile(s1, self.alpha)
        self.n_cal_hall, self.n_cal_ver = int(len(s0)), int(len(s1))
        self.fitted = bool(len(s0) and len(s1))
        return self

    def prediction_set(self, p: float) -> List[int]:
        out = []
        if p <= self.q_hall:          # 1 - p_0 = p  <= q_0
            out.append(0)
        if 1.0 - p <= self.q_ver:     # 1 - p_1      <= q_1
            out.append(1)
        return out

    def decide(self, p: float) -> str:
        if not self.fitted:
            return UNCALIBRATED
        s = self.decision_set(p)
        if s == [1]:
            return "VERIFIED"
        if s == [0]:
            return "HALLUCINATED"
        return "ABSTAIN"  # ambiguous: the data cannot decide at level alpha

    def decision_set(self, p: float) -> List[int]:
        """
        The conformal set, enlarged to {0,1} whenever its single class disagrees with the more probable class.
        Enlarging a prediction set can only increase coverage, so the per-class guarantee is preserved, and a
        VERIFIED verdict can never be shown next to P(hallucination) > 50% (or HALLUCINATED next to < 50%).
        """
        s = self.prediction_set(p)
        if s == [1] and p < 0.5 or s == [0] and p >= 0.5 or not s:
            return [0, 1]
        return s

    @property
    def implied_thresholds(self) -> Dict[str, float]:
        """For display only: VERIFIED iff p >= tau_ver (and p > q_hall); HALLUCINATED iff p <= tau_hall."""
        return {"tau_verified": round(1.0 - self.q_ver, 6), "tau_hallucinated": round(self.q_hall, 6)}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(self.implied_thresholds)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "MondrianConformal":
        if not d:
            return cls()
        keys = {"alpha", "q_hall", "q_ver", "n_cal_hall", "n_cal_ver", "fitted"}
        return cls(**{k: v for k, v in d.items() if k in keys})


def empirical_coverage(conf: MondrianConformal, p: Sequence[float], y: Sequence[int]) -> Dict[str, float]:
    """Class-conditional coverage, abstention rate and selective accuracy on a held-out set."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    sets = [conf.decision_set(pi) for pi in p]
    covered = np.array([yi in s for yi, s in zip(y, sets)])
    singleton = np.array([len(s) == 1 for s in sets])
    correct_committed = np.array([len(s) == 1 and s[0] == yi for yi, s in zip(y, sets)])
    out = {
        "coverage": float(covered.mean()) if len(y) else float("nan"),
        "coverage_supported": float(covered[y == 1].mean()) if np.any(y == 1) else float("nan"),
        "coverage_hallucinated": float(covered[y == 0].mean()) if np.any(y == 0) else float("nan"),
        "abstention_rate": float(1.0 - singleton.mean()) if len(y) else float("nan"),
        "selective_accuracy": float(correct_committed.sum() / max(1, singleton.sum())),
        "n": int(len(y)),
    }
    return out


@dataclass
class DecisionBundle:
    """Everything the evaluator needs at inference time, persisted in production_model_metrics.json."""
    schema_version: str = FEATURE_SCHEMA_VERSION
    pool: LogLinearPool = field(default_factory=LogLinearPool)
    conformal: MondrianConformal = field(default_factory=MondrianConformal)

    def to_dict(self) -> Dict:
        return {"schema_version": self.schema_version, "pool": self.pool.to_dict(), "conformal": self.conformal.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "DecisionBundle":
        if not d:
            return cls()
        return cls(
            schema_version=d.get("schema_version", "unknown"),
            pool=LogLinearPool.from_dict(d.get("pool")),
            conformal=MondrianConformal.from_dict(d.get("conformal")),
        )
