import tempfile
import subprocess
import os
import numpy as np

from ..schemas.analyze import FileRiskResult, SHAPBreakdown
from ..ml.git_extractor import get_git_stats
from ..ml.lizard_scanner import get_source_files, compute_complexity
from ..ml.helpers import percentile_rank
from ..ml.models import MLEngine

import logging

ml_engine = MLEngine()

logger = logging.getLogger(__name__)


class AnalyzerService:
    @staticmethod
    def analyze_repository(
        repo_url: str, branch: str | None = None, access_token: str | None = None
    ) -> tuple[list[FileRiskResult], float]:
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_url = repo_url
            if access_token:
                clone_url = clone_url.replace(
                    "https://", f"https://x-access-token:{access_token}@"
                )

            try:
                logger.info(f"Cloning repository into {temp_dir}...")
                clone_cmd = [
                    "git",
                    "clone",
                    "--shallow-since=6 months ago",
                    "--single-branch",
                ]
                if branch:
                    clone_cmd.extend(["--branch", branch])
                clone_cmd.extend([clone_url, temp_dir])

                subprocess.run(clone_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Git Clone Failed: {e.stderr.decode()}")

            # Git stats
            git_stats = get_git_stats(temp_dir)
            all_entropies = [stats.get("ent", 0.0) for stats in git_stats.values()]
            all_ndev = [stats.get("ndev", 0) for stats in git_stats.values()]
            all_age = [stats.get("age_days", 0.0) for stats in git_stats.values()]

            # Lizard stats
            source_files = get_source_files(temp_dir)
            raw_complexity = {}
            for fp in source_files:
                rel_path = os.path.relpath(fp, temp_dir)
                raw_complexity[rel_path] = compute_complexity(fp)

            all_ccn = [c["AvgCCN"] for c in raw_complexity.values()]
            all_nloc = [c["AvgNLOC"] for c in raw_complexity.values()]

            results = []

            # Score every file
            for fp in source_files:
                rel = os.path.relpath(fp, temp_dir)
                stats = git_stats.get(rel, {})
                msgs = stats.get("messages", [])
                n_commits = stats.get("nf", 0)

                # Low confidence flag for files with minimal history
                low_confidence = n_commits < 10

                # Sentiment
                sentiment_score, low_info_ratio, commit_sentiments = (
                    ml_engine.compute_sentiment(msgs)
                )

                # Complexity Interaction
                # Notice we use the percentile ranks to isolate true bloat!
                ccn_score = percentile_rank(raw_complexity[rel]["AvgCCN"], all_ccn)
                nloc_score = percentile_rank(raw_complexity[rel]["AvgNLOC"], all_nloc)
                complexity_score = (ccn_score * nloc_score) ** 0.5

                # Risk Fusion
                change_entropy = percentile_rank(stats.get("ent", 0.0), all_entropies)
                ndev_score = percentile_rank(stats.get("ndev", 0), all_ndev)
                age_score = percentile_rank(stats.get("age_days", 0.0), all_age)

                X_file_raw = np.array(
                    [
                        [
                            sentiment_score,
                            change_entropy,
                            ndev_score,
                            age_score,
                            complexity_score,
                            low_info_ratio,
                            float(n_commits),
                        ]
                    ]
                )

                X_file_scaled = ml_engine.scaler.transform(X_file_raw)

                risk_score = float(
                    ml_engine.risk_model.predict_proba(X_file_scaled)[0][1]
                )

                # SHAP Explanations
                shap_values = ml_engine.shap_explainer.shap_values(X_file_scaled)

                sv = (
                    shap_values[1][0]
                    if isinstance(shap_values, list)
                    else shap_values[0]
                )
                base_val = ml_engine.shap_explainer.expected_value
                base = float(base_val[1] if hasattr(base_val, "__len__") else base_val)

                breakdown = SHAPBreakdown(
                    base_rate=base,
                    sentiment_contrib=float(sv[0]),
                    entropy_contrib=float(sv[1]),
                    ndev_contrib=float(sv[2]),
                    age_contrib=float(sv[3]),
                    complexity_contrib=float(sv[4]),
                    low_info_contrib=float(sv[5]),
                    commits_contrib=float(sv[6]),
                )

                results.append(
                    FileRiskResult(
                        filename=rel,
                        risk_score=risk_score,
                        sentiment_score=sentiment_score,
                        complexity_score=complexity_score,
                        low_confidence=low_confidence,
                        shap_breakdown=breakdown,
                        commit_sentiments=commit_sentiments,
                    )
                )

            # Sort so the riskiest files are at the top
            results.sort(key=lambda x: x.risk_score, reverse=True)
            return results, ml_engine.threshold
