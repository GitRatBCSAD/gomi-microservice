import json
from typing import Any
import numpy as np
import joblib
import shap
from transformers import (
    pipeline,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from ..schemas.analyze import CommitSentiment

import logging

logger = logging.getLogger(__name__)


class MLEngine:
    sentiment_pipeline: Any
    risk_model: Any
    scaler: Any
    shap_explainer: Any
    threshold: float

    def __init__(self):
        self.sentiment_pipeline = None
        self.risk_model = None
        self.scaler = None
        self.shap_explainer = None
        self.threshold = 0.5

    def load_models(
        self, hf_sentiment_repo: str, hf_risk_repo: str, hf_token: str | None = None
    ):
        """
        Loads the Hugging Face DistilBERT model, and downloads the scikit-learn models from Hugging Face.
        Run this ONCE when the FastAPI server starts.
        """
        logger.info("Loading DistilBERT Sentiment Model...")
        # Using Hugging Face's pipeline makes tokenization and softmax prediction a 1-liner
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            hf_sentiment_repo, token=hf_token
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            hf_sentiment_repo, token=hf_token
        )
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=128,
        )

        logger.info("Downloading Risk Fusion Model from Hugging Face...")
        from huggingface_hub import hf_hub_download

        # Download the files directly from the HF Hub
        risk_model_path = hf_hub_download(
            repo_id=hf_risk_repo, filename="risk_model.joblib", token=hf_token
        )
        scaler_path = hf_hub_download(
            repo_id=hf_risk_repo, filename="gomi_scaler.joblib", token=hf_token
        )
        shap_path = hf_hub_download(
            repo_id=hf_risk_repo,
            filename="risk_model_shap_background.npy",
            token=hf_token,
        )
        threshold_path = hf_hub_download(
            repo_id=hf_risk_repo, filename="risk_threshold.json", token=hf_token
        )

        # Load the downloaded sklearn objects
        self.risk_model = joblib.load(risk_model_path)
        self.scaler = joblib.load(scaler_path)

        # Load SHAP with the training background distribution
        X_train_bg = np.load(shap_path)
        self.shap_explainer = shap.LinearExplainer(self.risk_model, X_train_bg)

        with open(threshold_path) as f:
            self.threshold = float(json.load(f)["threshold"])

        logger.info(f"All models loaded. F1-optimal threshold: {self.threshold:.4f}")

    def compute_sentiment(
        self, commits: list[dict[str, Any]]
    ) -> tuple[float, float, list[CommitSentiment]]:
        """
        Evaluates a list of commits: [{hash, message, committed_at}, ...].
        Low-info messages are excluded from classification entirely (matches
        training -- DistilBERT is never run on them, see gomi_train.ipynb),
        not assigned a placeholder label.
        Returns: (sentiment_score, low_info_ratio, list_of_commit_sentiments)
        """
        if not commits:
            return 0.0, 0.0, []

        low_info_commits = [c for c in commits if len(c["message"].split()) < 5]
        valid_commits = [c for c in commits if len(c["message"].split()) >= 5]
        low_info_ratio = len(low_info_commits) / len(commits)

        commit_sentiments: list[CommitSentiment] = []
        for c in low_info_commits:
            commit_sentiments.append(
                CommitSentiment(
                    hash=c["hash"],
                    message=c["message"],
                    committed_at=c["committed_at"],
                    code=None,
                    low_info=True,
                    risk_probability=None,
                )
            )

        if not valid_commits:
            return 0.0, low_info_ratio, commit_sentiments

        # Batch predict sentiment for the valid messages
        valid_msgs = [c["message"] for c in valid_commits]
        results = self.sentiment_pipeline(valid_msgs)

        # Calculate ratio of "risky" emotions and save the messages
        # TODO: risk_probability is left unset here -- the pipeline call above
        # only returns the top-1 label, not the full score distribution, so a
        # real p(caution) isn't available yet. Needs top_k=None (see fix #3).
        risk_count = 0
        for c, r in zip(valid_commits, results):
            label = r["label"].lower()
            commit_sentiments.append(
                CommitSentiment(
                    hash=c["hash"],
                    message=c["message"],
                    committed_at=c["committed_at"],
                    code=label,
                    low_info=False,
                    risk_probability=None,
                )
            )
            if label == "caution":
                risk_count += 1

        sentiment_score = risk_count / len(valid_commits)

        return sentiment_score, low_info_ratio, commit_sentiments
