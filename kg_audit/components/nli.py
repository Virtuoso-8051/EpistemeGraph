"""
Natural-language-inference verifier (claim  vs.  evidence sentence).

A cross-encoder (DeBERTa-v3-xsmall fine-tuned on SNLI+MNLI) reads the premise (a sentence from the
source papers) and the hypothesis (a verbalised claim) jointly and returns
    P(entailment), P(contradiction), P(neutral).
Unlike cosine similarity of independent embeddings, NLI distinguishes "says the same thing" from
"talks about the same thing", and detects negation and swapped relations. This is the backbone of
modern factual-consistency metrics (SummaC, AlignScore, FActScore's NLI variant).

Runs on CPU (~0.1 s per claim with top-k=3 evidence), loaded once per process, no API cost.
If the model cannot be loaded, `available` is False and callers fall back to neutral values
(and the evaluation report records it, so a training/inference mismatch is visible).
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import numpy as np

from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import limit_cpu_threads

NLI_MODEL = os.getenv("EPISTEME_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")


class NLIVerifier:
    _instance: Optional["NLIVerifier"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name
        self.model = None
        self.idx = {"contradiction": 0, "entailment": 1, "neutral": 2}
        if os.getenv("EPISTEME_DISABLE_NLI") == "1":
            return
        try:
            limit_cpu_threads()
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device="cpu")
            labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
            self.idx = {k: labels[k] for k in ("contradiction", "entailment", "neutral")}
        except Exception as e:  # offline / not downloaded / incompatible
            logging.warning(f"NLI model unavailable ({e}); NLI features fall back to neutral values")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @classmethod
    def get(cls) -> "NLIVerifier":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def score(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """pairs = [(premise, hypothesis)] -> array (n, 2) of [P(entail), P(contradict)]."""
        if not pairs:
            return np.zeros((0, 2))
        if not self.available:
            return np.tile([1 / 3, 1 / 3], (len(pairs), 1))
        p = np.asarray(self.model.predict(pairs, apply_softmax=True, batch_size=batch_size, show_progress_bar=False))
        return np.column_stack([p[:, self.idx["entailment"]], p[:, self.idx["contradiction"]]])

    def verify_claims(self, claims: List[str], evidence: List[List[str]]) -> np.ndarray:
        """
        For claim i with candidate evidence sentences E_i, returns columns
            entail_i = max_{s in E_i} P(entail | s, c_i)
            contra_i = max_{s in E_i} P(contradict | s, c_i)
            cons_i   = max_{s in E_i} [P(entail | s, c_i) - P(contradict | s, c_i)]   in [-1, 1]
        cons_i is the zero-shot SummaC consistency score (Laban et al., 2022): taking the difference
        PER PREMISE before the max means an unrelated sentence that happens to "contradict" a short
        claim cannot dominate, which a separate max-contradiction would allow.
        Returns array (n_claims, 3).
        """
        flat, owner = [], []
        for i, (c, ev) in enumerate(zip(claims, evidence)):
            for s in ev:
                if s:
                    flat.append((s, c))
                    owner.append(i)
        out = np.zeros((len(claims), 3))
        out[:, 2] = -1.0
        if not flat:
            out[:, 2] = 0.0
            return out
        sc = self.score(flat)
        for (e, x), i in zip(sc, owner):
            out[i, 0] = max(out[i, 0], e)
            out[i, 1] = max(out[i, 1], x)
            out[i, 2] = max(out[i, 2], e - x)
        return out
