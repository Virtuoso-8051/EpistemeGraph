import os
import sys
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
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


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config = config
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

    def generate_synthetic_benchmark_dataset(self, n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=42)
        half = n_samples // 2

        X_pos = np.column_stack([
            rng.uniform(0.01, 0.40, half),
            rng.uniform(0.00, 0.45, half),
            rng.uniform(0.01, 0.35, half),
            rng.uniform(0.65, 0.98, half),
            rng.uniform(0.60, 1.00, half),
            rng.uniform(0.55, 1.00, half),
            rng.uniform(0.00, 0.20, half),
            rng.uniform(0.70, 0.99, half),
        ])
        y_pos = np.ones(half, dtype=int)

        X_neg = np.column_stack([
            rng.uniform(0.40, 2.50, half),
            rng.uniform(0.45, 2.50, half),
            rng.uniform(0.30, 1.80, half),
            rng.uniform(0.20, 0.85, half),
            rng.uniform(0.20, 0.95, half),
            rng.uniform(0.15, 0.85, half),
            rng.uniform(0.20, 1.00, half),
            rng.uniform(0.70, 0.99, half),
        ])
        y_neg = np.zeros(half, dtype=int)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

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
            logging.info("Starting GridSearchCV + Calibrated Belief ModelTrainer across 7 ML/DL architectures...")

            if X_train is None or y_train is None:
                X_all, y_all = self.generate_synthetic_benchmark_dataset(n_samples=800)
                split_idx = int(0.8 * len(X_all))
                X_train, X_test = X_all[:split_idx], X_all[split_idx:]
                y_train, y_test = y_all[:split_idx], y_all[split_idx:]

            models_and_grids: Dict[str, Tuple[Any, Dict[str, Any]]] = {
                "LogisticRegression": (
                    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=42))]),
                    {"clf__C": [0.1, 1.0, 10.0]}
                ),
                "RandomForest": (
                    RandomForestClassifier(random_state=42),
                    {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 10]}
                ),
                "ExtraTrees": (
                    ExtraTreesClassifier(random_state=42),
                    {"n_estimators": [50, 100], "max_depth": [6, 10]}
                ),
                "GradientBoosting": (
                    GradientBoostingClassifier(random_state=42),
                    {"n_estimators": [50, 100], "learning_rate": [0.03, 0.1], "max_depth": [3, 5]}
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
                    {"svm__C": [0.5, 1.0, 5.0], "svm__gamma": ["scale", "auto"]}
                )
            }

            leaderboard = {}
            best_model_name = None
            best_calibrated_model = None
            best_composite_score = -np.inf

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

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_calibrated_model, f)

            report_payload = {
                "winning_model": best_model_name,
                "is_promoted_to_production": is_accepted,
                "leaderboard": leaderboard
            }
            with open(self.config.metrics_report_file_path, "w") as f:
                json.dump(report_payload, f, indent=4)

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