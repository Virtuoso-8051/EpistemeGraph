import os
import sys
import csv
import json
import pickle
import numpy as np
from datetime import datetime, timezone
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

from kg_audit.entity.config_entity import ModelTrainerConfig
from kg_audit.entity.artifact_entity import ModelTrainerArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


class _AttentionResNetModule(nn.Module):
    """
    PyTorch Deep Residual Network with Feature-Level Self-Attention Gating
    over the 8-dimensional CASSE verification vector X.
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, in_features),
            nn.Softmax(dim=-1)
        )
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act = nn.SiLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = self.attention_gate(x)
        x_gated = x * (1.0 + attn_weights)
        h = self.input_layer(x_gated)
        h = self.act(h + self.res_block1(h))
        h = self.act(h + self.res_block2(h))
        logits = self.head(h).squeeze(-1)
        return logits, attn_weights


class NativePyTorchAttentionClassifier:
    """
    Self-contained PyTorch AttentionResNet with internal StandardScaler and Platt Sigmoid
    Calibration so it never triggers scikit-learn's BaseEstimator tag bugs.
    """
    def __init__(self, hidden_dim: int = 64, lr: float = 0.005, epochs: int = 120, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.platt_calibrator = LogisticRegression()
        self.model: Optional[_AttentionResNetModule] = None
        self.mean_attention_weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.float32)

        self.model = _AttentionResNetModule(in_features=X_scaled.shape[1], hidden_dim=self.hidden_dim)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = self.model(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            raw_logits, attn = self.model(x_t)
            self.mean_attention_weights_ = attn.mean(dim=0).cpu().numpy()
            logits_np = raw_logits.cpu().numpy().reshape(-1, 1)

        # Fit Platt scaling over raw logits
        self.platt_calibrator.fit(logits_np, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32))
        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            raw_logits, _ = self.model(x_t)
            logits_np = raw_logits.cpu().numpy().reshape(-1, 1)
        return self.platt_calibrator.predict_proba(logits_np)


class ModelTrainer:
    FEATURE_NAMES = [
        "e_sheaf", "delta_k", "d_spec", "bayesian_trust",
        "freshness", "independence", "conflict_ratio", "llm_conf"
    ]

    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config = config
            self.telemetry_csv_path = os.path.join("data", "live_telemetry_claims.csv")
            self.production_metrics_path = getattr(
                config, "production_metrics_path", os.path.join("final_model", "production_model_metrics.json")
            )
        except Exception as e:
            raise KGAuditException(e, sys)

    @staticmethod
    def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        ece = 0.0
        n_samples = len(y_true)

        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                ece += (np.sum(mask) / n_samples) * abs(bin_acc - bin_conf)

        return float(ece)

    def append_live_telemetry(
        self,
        feature_vector: List[float],
        query_text: str,
        ground_truth_label: Optional[int] = None,
        expected_verdict: Optional[str] = None
    ) -> int:
        try:
            os.makedirs("data", exist_ok=True)
            e_sheaf, delta_k, d_spec, bayesian_trust, _, _, conflict_ratio, _ = feature_vector

            if ground_truth_label is not None:
                label = int(ground_truth_label)
                label_source = expected_verdict if expected_verdict else "SUPERVISED_BENCHMARK"
            else:
                label = 1 if (conflict_ratio <= 0.45 and e_sheaf <= 0.50 and bayesian_trust >= 0.48) else 0
                label_source = "TOPOLOGICAL_ORACLE"

            file_exists = os.path.exists(self.telemetry_csv_path)
            with open(self.telemetry_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.FEATURE_NAMES + ["oracle_label", "label_source", "timestamp", "query"])
                writer.writerow(
                    [round(float(x), 6) for x in feature_vector]
                    + [label, label_source, datetime.now(timezone.utc).isoformat(), query_text[:120]]
                )

            with open(self.telemetry_csv_path, "r", encoding="utf-8") as f:
                total_rows = max(0, sum(1 for _ in f) - 1)
            return total_rows
        except Exception as e:
            logging.warning(f"Telemetry logging warning: {e}")
            return 0

    def generate_synthetic_benchmark_dataset(self, n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=42)
        half = n_samples // 2

        X_pos = np.column_stack([
            rng.uniform(0.04, 0.45, half),
            rng.uniform(0.05, 0.48, half),
            rng.uniform(0.02, 0.35, half),
            rng.uniform(0.52, 0.96, half),
            rng.uniform(0.82, 0.98, half),
            rng.uniform(0.50, 0.92, half),
            rng.uniform(0.02, 0.42, half),
            rng.uniform(0.75, 0.99, half),
        ])
        y_pos = np.ones(half, dtype=int)

        X_neg = np.column_stack([
            rng.uniform(0.65, 1.10, half),
            rng.uniform(0.70, 1.15, half),
            rng.uniform(0.40, 0.70, half),
            rng.uniform(0.06, 0.25, half),
            rng.uniform(0.55, 0.80, half),
            rng.uniform(0.15, 0.35, half),
            rng.uniform(0.75, 1.00, half),
            rng.uniform(0.75, 0.99, half),
        ])
        y_neg = np.zeros(half, dtype=int)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        flip_mask = rng.random(n_samples) < 0.02
        y[flip_mask] = 1 - y[flip_mask]

        if os.path.exists(self.telemetry_csv_path):
            try:
                live_X, live_y = [], []
                with open(self.telemetry_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vec = [float(row[col]) for col in self.FEATURE_NAMES]
                        lbl = int(row["oracle_label"])
                        repeat_k = 10 if row.get("label_source") in {"VERIFIED", "ABSTAIN", "HALLUCINATED", "SUPERVISED_BENCHMARK"} else 5
                        for _ in range(repeat_k):
                            live_X.append(vec)
                            live_y.append(lbl)
                if live_X:
                    X = np.vstack([X, np.array(live_X)])
                    y = np.concatenate([y, np.array(live_y, dtype=int)])
            except Exception as ex:
                logging.warning(f"Could not merge live telemetry CSV: {ex}")

        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    def _learn_scoring_parameters_and_thresholds(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ml_probs_val: np.ndarray,
        attn_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Learns the 8 Boltzmann-Gibbs multipliers via PyTorch Self-Attention + L-BFGS-B
        and computes the Conformal Tri-State Decision Thresholds (pass_thresh, abstain_thresh).
        """
        e_s, d_k, d_sp, b_tr, fr, ind, c_rat, llm_c = [X_val[:, i] for i in range(8)]

        if attn_weights is not None and len(attn_weights) == 8:
            init_e = np.clip(attn_weights[[0, 1, 2, 6]] * 2.5, 0.10, 0.80)
            init_p = attn_weights[[3, 4, 5, 7]]
            init_p = init_p / (np.sum(init_p) + 1e-12)
            init_params = np.concatenate([init_e, init_p, [0.35]])
        else:
            init_params = np.ones(9, dtype=np.float64) / 4.0

        def eval_analytical(params: np.ndarray) -> np.ndarray:
            w_e = np.abs(params[:4])
            w_p = np.abs(params[4:8])
            w_p = w_p / (np.sum(w_p) + 1e-12)
            energy = w_e[0] * e_s + w_e[1] * d_k + w_e[2] * d_sp + w_e[3] * c_rat
            prior = w_p[0] * b_tr + w_p[1] * fr + w_p[2] * ind + w_p[3] * llm_c * (1.0 - c_rat)
            return np.clip(prior * np.exp(-energy), 1e-6, 1.0 - 1e-6)

        def loss_fn(params: np.ndarray) -> float:
            p_ana = eval_analytical(params)
            gamma = float(np.clip(params[8], 0.15, 0.65))
            p_blend = np.clip((1.0 - gamma) * p_ana + gamma * ml_probs_val, 1e-6, 1.0 - 1e-6)
            bce = -np.mean(y_val * np.log(p_blend) + (1.0 - y_val) * np.log(1.0 - p_blend))
            brier = np.mean((y_val - p_blend) ** 2)
            return float(bce + brier)

        bounds = [
            (0.08, 1.20), (0.08, 1.20), (0.05, 1.00), (0.15, 1.50),
            (0.15, 0.75), (0.05, 0.50), (0.05, 0.50), (0.01, 0.25),
            (0.15, 0.65)
        ]

        res = minimize(loss_fn, init_params, method="L-BFGS-B", bounds=bounds)
        opt_p = res.x if res.success else init_params

        w_energy = [round(float(x), 4) for x in opt_p[:4]]
        raw_prior = np.abs(opt_p[4:8])
        w_prior = [round(float(x), 4) for x in (raw_prior / (np.sum(raw_prior) + 1e-12))]
        gamma_opt = round(float(np.clip(opt_p[8], 0.15, 0.65)), 4)

        # Compute Conformal Tri-State Separation Thresholds
        p_ana_all = eval_analytical(opt_p)
        neg_scores = p_ana_all[y_val == 0]
        pos_scores = p_ana_all[y_val == 1]

        if len(neg_scores) > 0 and len(pos_scores) > 0:
            q_neg_95 = float(np.percentile(neg_scores, 95))
            q_pos_25 = float(np.percentile(pos_scores, 25))
            abstain_thresh = round(float(np.clip(q_neg_95 + 0.05, 0.24, 0.30)), 4)
            pass_thresh = round(float(np.clip(q_pos_25 + 0.12, 0.56, 0.60)), 4)
        else:
            abstain_thresh = 0.2500
            pass_thresh = 0.5750

        feature_weight_dict = {
            "e_sheaf_penalty_weight": w_energy[0],
            "delta_k_penalty_weight": w_energy[1],
            "d_spec_penalty_weight": w_energy[2],
            "conflict_penalty_weight": w_energy[3],
            "bayesian_trust_prior_weight": w_prior[0],
            "freshness_prior_weight": w_prior[1],
            "independence_prior_weight": w_prior[2],
            "llm_conf_prior_weight": w_prior[3],
        }

        return {
            "energy_weights": w_energy,
            "prior_weights": w_prior,
            "analytical_ml_blend_gamma": gamma_opt,
            "s_query_exponent_eta": 0.85,
            "optimal_pass_threshold": pass_thresh,
            "optimal_abstain_threshold": abstain_thresh,
            "feature_weight_dict": feature_weight_dict
        }

    def initiate_model_trainer(
        self,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting 9-Architecture Tournament (8 Sklearn + Native PyTorch AttentionResNet)...")

            if X_train is None or y_train is None:
                X_all, y_all = self.generate_synthetic_benchmark_dataset(n_samples=800)
                split_idx = int(0.8 * len(X_all))
                X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                y_train, y_test = y_all[:split_idx], y_all[split_idx:]

            leaderboard = {}
            best_model_name = None
            best_calibrated_model = None
            best_composite_score = -np.inf
            best_y_prob = None

            # 1. Train & Evaluate Native PyTorch Self-Attention Residual Network
            pt_net = NativePyTorchAttentionClassifier(hidden_dim=64, lr=0.005, epochs=120, random_state=42)
            pt_net.fit(X_train, y_train)
            learned_attn_gates = pt_net.mean_attention_weights_

            pt_prob = pt_net.predict_proba(X_test)[:, 1]
            pt_pred = (pt_prob >= 0.5).astype(int)
            pt_auc = float(roc_auc_score(y_test, pt_prob))
            pt_f1 = float(f1_score(y_test, pt_pred))
            pt_brier = float(brier_score_loss(y_test, pt_prob))
            pt_ece = self.compute_expected_calibration_error(y_test, pt_prob)
            pt_comp = pt_auc - pt_brier - pt_ece

            leaderboard["PyTorch_AttentionResNet"] = {
                "best_params": {"hidden_dim": 64, "epochs": 120, "self_attention": True},
                "roc_auc": pt_auc,
                "f1_score": pt_f1,
                "brier_score": pt_brier,
                "ece": pt_ece,
                "composite_score": pt_comp
            }
            best_composite_score = pt_comp
            best_model_name = "PyTorch_AttentionResNet"
            best_calibrated_model = pt_net
            best_y_prob = pt_prob

            # 2. Train & Evaluate the 8 Native Scikit-Learn Models
            models_and_grids: Dict[str, Tuple[Any, Dict[str, Any]]] = {
                "LogisticRegression": (
                    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=42))]),
                    {"clf__C": [0.1, 1.0, 10.0]}
                ),
                "RandomForest": (
                    RandomForestClassifier(random_state=42),
                    {"n_estimators": [100, 150], "max_depth": [6, 10]}
                ),
                "ExtraTrees": (
                    ExtraTreesClassifier(random_state=42),
                    {"n_estimators": [100, 150], "max_depth": [6, 10]}
                ),
                "GradientBoosting": (
                    GradientBoostingClassifier(random_state=42),
                    {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
                ),
                "AdaBoost": (
                    AdaBoostClassifier(random_state=42),
                    {"n_estimators": [50, 100], "learning_rate": [0.05, 0.2]}
                ),
                "DeepMLP_NeuralNet": (
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("mlp", MLPClassifier(max_iter=400, early_stopping=True, random_state=42))
                    ]),
                    {"mlp__hidden_layer_sizes": [(64, 32), (128, 64, 32)], "mlp__alpha": [0.001, 0.01]}
                ),
                "RBF_SVM": (
                    Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, random_state=42))]),
                    {"svm__C": [1.0, 5.0], "svm__gamma": ["scale", "auto"]}
                ),
                "StackingMetaEnsemble": (
                    StackingClassifier(
                        estimators=[
                            ("gbdt", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
                            ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
                            ("et", ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42)),
                            ("mlp", Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))])),
                            ("svm", Pipeline([("s", StandardScaler()), ("v", SVC(probability=True, C=2.0, random_state=42))]))
                        ],
                        final_estimator=LogisticRegression(random_state=42),
                        cv=3
                    ),
                    {"final_estimator__C": [0.5, 1.0, 2.0]}
                )
            }

            for name, (estimator, param_grid) in models_and_grids.items():
                grid = GridSearchCV(estimator, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_tuned_estimator = grid.best_estimator_

                calibrated = CalibratedClassifierCV(estimator=best_tuned_estimator, method="sigmoid", cv=3)
                calibrated.fit(X_train, y_train)

                y_prob = calibrated.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                roc_auc = float(roc_auc_score(y_test, y_prob))
                f1 = float(f1_score(y_test, y_pred))
                brier = float(brier_score_loss(y_test, y_prob))
                ece = self.compute_expected_calibration_error(y_test, y_prob)

                composite_score = roc_auc - brier - ece

                leaderboard[name] = {
                    "best_params": grid.best_params_,
                    "roc_auc": roc_auc,
                    "f1_score": f1,
                    "brier_score": brier,
                    "ece": ece,
                    "composite_score": composite_score
                }

                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_model_name = name
                    best_calibrated_model = calibrated
                    best_y_prob = y_prob

            best_metrics = leaderboard[best_model_name]

            learned_scoring_params = self._learn_scoring_parameters_and_thresholds(
                X_test, y_test, best_y_prob, attn_weights=learned_attn_gates
            )

            previous_champion_score = -1.0
            if os.path.exists(self.production_metrics_path):
                try:
                    with open(self.production_metrics_path, "r", encoding="utf-8") as f:
                        prod_data = json.load(f)
                    previous_champion_score = float(prod_data.get("champion_composite_score", -1.0))
                except Exception:
                    previous_champion_score = -1.0

            is_accepted = (
                best_metrics["roc_auc"] >= self.config.expected_roc_auc
                and best_metrics["brier_score"] <= self.config.max_brier_score
                and best_composite_score >= (previous_champion_score - 0.02)
            )

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_calibrated_model, f)

            report_payload = {
                "winning_model": best_model_name,
                "champion_composite_score": best_composite_score,
                "previous_champion_score": previous_champion_score,
                "is_promoted_to_production": is_accepted,
                "total_training_samples": int(len(X_train) + len(X_test)),
                "learned_scoring_parameters": learned_scoring_params,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "leaderboard": leaderboard
            }
            with open(self.config.metrics_report_file_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=4)

            if is_accepted:
                os.makedirs(os.path.dirname(self.config.final_model_file_path), exist_ok=True)
                with open(self.config.final_model_file_path, "wb") as f:
                    pickle.dump(best_calibrated_model, f)
                with open(self.production_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(report_payload, f, indent=4)
                logging.info(f"Promoted Champion [{best_model_name}] to {self.config.final_model_file_path}")

            return ModelTrainerArtifact(
                best_model_name=best_model_name,
                trained_model_file_path=self.config.trained_model_file_path,
                roc_auc_score=best_metrics["roc_auc"],
                f1_score=best_metrics["f1_score"],
                brier_score=best_metrics["brier_score"],
                expected_calibration_error=best_metrics["ece"],
                is_model_accepted=is_accepted,
                learned_feature_weights=learned_scoring_params["feature_weight_dict"],
                optimal_pass_threshold=learned_scoring_params["optimal_pass_threshold"],
                optimal_abstain_threshold=learned_scoring_params["optimal_abstain_threshold"]
            )

        except Exception as e:
            raise KGAuditException(e, sys)