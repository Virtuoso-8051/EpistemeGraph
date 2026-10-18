import os
import sys
import json
import pickle
import numpy as np
import mlflow
from typing import Tuple, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

from kg_audit.entity.config_entity import ModelTrainerConfig
from kg_audit.entity.artifact_entity import ModelTrainerArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config = config
        except Exception as e:
            raise KGAuditException(e, sys)

    @staticmethod
    def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE): measures how closely the predicted
        belief probabilities match empirical accuracy across confidence bins.
        """
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

    def generate_synthetic_benchmark_dataset(self, n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a controlled 8-feature benchmark dataset of Verified (1) vs. Hallucinated (0)
        KG claims for bootstrapping and testing the ModelTrainer pipeline before live logs accumulate.
        Features: [e_sheaf, delta_k, d_spec, bayesian_trust, freshness, independence, conflict_ratio, llm_conf]
        """
        rng = np.random.default_rng(seed=42)
        half = n_samples // 2

        # Class 1: Factual / Verified Claims (Low topological energy, high source trust, low conflict)
        X_pos = np.column_stack([
            rng.uniform(0.01, 0.35, half),  # e_sheaf
            rng.uniform(0.00, 0.40, half),  # delta_k
            rng.uniform(0.01, 0.30, half),  # d_spec
            rng.uniform(0.70, 0.98, half),  # bayesian_trust
            rng.uniform(0.65, 1.00, half),  # freshness
            rng.uniform(0.60, 1.00, half),  # independence
            rng.uniform(0.00, 0.15, half),  # conflict_ratio
            rng.uniform(0.75, 0.99, half),  # llm_conf
        ])
        y_pos = np.ones(half, dtype=int)

        # Class 0: Hallucinated Claims (High sheaf/curvature distortion or high conflict, even if llm_conf is high)
        X_neg = np.column_stack([
            rng.uniform(0.45, 2.50, half),  # e_sheaf
            rng.uniform(0.50, 2.50, half),  # delta_k
            rng.uniform(0.35, 1.80, half),  # d_spec
            rng.uniform(0.20, 0.85, half),  # bayesian_trust
            rng.uniform(0.20, 0.95, half),  # freshness
            rng.uniform(0.15, 0.80, half),  # independence
            rng.uniform(0.25, 1.00, half),  # conflict_ratio
            rng.uniform(0.70, 0.99, half),  # llm_conf (LLMs are often overconfident when hallucinating!)
        ])
        y_neg = np.zeros(half, dtype=int)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        # Add 5% boundary noise so classification is realistic and non-trivial
        flip_mask = rng.random(n_samples) < 0.05
        y[flip_mask] = 1 - y[flip_mask]

        perm = rng.permutation(n_samples)
        return X[perm], y[perm]

    def initiate_model_trainer(
        self,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Calibrated Belief ModelTrainer benchmark...")

            if X_train is None or y_train is None:
                X_all, y_all = self.generate_synthetic_benchmark_dataset(n_samples=600)
                split_idx = int(0.8 * len(X_all))
                X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                y_train, y_test = y_all[:split_idx], y_all[split_idx:]

            candidate_models: Dict[str, Any] = {
                "LogisticRegression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=80, learning_rate=0.1, random_state=42)
            }

            leaderboard = {}
            best_model_name = None
            best_calibrated_model = None
            best_composite_score = -np.inf

            for name, base_model in candidate_models.items():
                # Wrap every classifier in Sigmoid (Platt) Cross-Validated Probability Calibration
                calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
                calibrated.fit(X_train, y_train)

                y_prob = calibrated.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                roc_auc = float(roc_auc_score(y_test, y_prob))
                f1 = float(f1_score(y_test, y_pred))
                brier = float(brier_score_loss(y_test, y_prob))
                ece = self.compute_expected_calibration_error(y_test, y_prob)

                # Composite selection metric: reward high ROC-AUC, penalize high Brier loss & ECE
                composite_score = roc_auc - brier - ece

                leaderboard[name] = {
                    "roc_auc": roc_auc,
                    "f1_score": f1,
                    "brier_score": brier,
                    "ece": ece,
                    "composite_score": composite_score
                }

                logging.info(
                    f"Model [{name}] -> ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}"
                )

                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_model_name = name
                    best_calibrated_model = calibrated

            best_metrics = leaderboard[best_model_name]
            is_accepted = (
                best_metrics["roc_auc"] >= self.config.expected_roc_auc
                and best_metrics["brier_score"] <= self.config.max_brier_score
            )

            # Save artifact in timestamped run folder
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_calibrated_model, f)

            # Save full comparative leaderboard JSON for research paper tables
            report_payload = {
                "winning_model": best_model_name,
                "is_promoted_to_production": is_accepted,
                "leaderboard": leaderboard
            }
            with open(self.config.metrics_report_file_path, "w") as f:
                json.dump(report_payload, f, indent=4)

            # Auto-Deploy Gatekeeper: only overwrite final_model/model.pkl if accepted
            if is_accepted:
                os.makedirs(os.path.dirname(self.config.final_model_file_path), exist_ok=True)
                with open(self.config.final_model_file_path, "wb") as f:
                    pickle.dump(best_calibrated_model, f)
                logging.info(f"Promoted {best_model_name} to {self.config.final_model_file_path}")

            return ModelTrainerArtifact(
                best_model_name=best_model_name,
                trained_model_file_path=self.config.trained_model_file_path,
                roc_auc_score=best_metrics["roc_auc"],
                f1_score=best_metrics["f1_score"],
                brier_score=best_metrics["brier_score"],
                expected_calibration_error=best_metrics["ece"],
                is_model_accepted=is_accepted
            )

        except Exception as e:
            raise KGAuditException(e, sys)