import numpy as np
import joblib
import shap
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification

class MLEngine:
    def __init__(self):
        self.sentiment_pipeline = None
        self.risk_model = None
        self.scaler = None
        self.shap_explainer = None

    def load_models(self, hf_repo_id: str, local_risk_dir: str):
        """
        Loads the Hugging Face DistilBERT model, and the local scikit-learn models.
        Run this ONCE when the FastAPI server starts.
        """
        print("Loading DistilBERT Sentiment Model...")

        tokenizer = DistilBertTokenizerFast.from_pretrained(hf_repo_id)
        model = DistilBertForSequenceClassification.from_pretrained(hf_repo_id)
        self.sentiment_pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            truncation=True, 
            max_length=128
        )

        print("Loading Risk Fusion Model (Logistic Regression)...")

        self.risk_model = joblib.load(f"{local_risk_dir}/risk_model.joblib")
        self.scaler = joblib.load(f"{local_risk_dir}/gomi_scaler.joblib")
        
        X_train_bg = np.load(f"{local_risk_dir}/risk_model_shap_background.npy")
        self.shap_explainer = shap.LinearExplainer(self.risk_model, X_train_bg)
        print("All models loaded successfully!")

    def compute_sentiment(self, messages: list[str]) -> tuple[float, float]:
        """
        Evaluates a list of commit messages.
        Returns: (sentiment_score, low_info_ratio)
        """
        if not messages:
            return 0.0, 0.0

        # Filter out low info messages 
        low_info_msgs = [m for m in messages if len(m.split()) < 5]
        low_info_ratio = len(low_info_msgs) / len(messages)

        valid_msgs = [m for m in messages if len(m.split()) >= 5]
        if not valid_msgs:
            return 0.0, low_info_ratio

        # Batch predict sentiment for the valid messages
        results = self.sentiment_pipeline(valid_msgs)
        
        # Calculate ratio of "risky" emotions
        risk_count = sum(1 for r in results if r['label'].lower() in ['frustration', 'caution'])
        sentiment_score = risk_count / len(valid_msgs)

        return sentiment_score, low_info_ratio
