import tempfile
import subprocess
import os
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed

from ..schemas.analyze import FileRiskResult, SHAPBreakdown, CommitSentiment
from ..ml.git_extractor import get_git_stats
from ..ml.lizard_scanner import get_source_files, compute_complexity
from ..ml.helpers import percentile_rank
from ..ml.models import MLEngine
from ..ml.constants import LOW_CONFIDENCE_COMMIT_THRESHOLD, MIN_COMMIT_WORDS

import logging

ml_engine = MLEngine()

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    file_results: list[FileRiskResult]
    threshold: float
    head_sha: str
    default_branch: str


class AnalyzerService:
    @staticmethod
    def analyze_repository(
        repo_url: str,
        branch: str | None = None,
        access_token: str | None = None,
        on_progress: Callable[[str, float, str | None], None] | None = None,
    ) -> AnalysisResult:
        def emit(step: str, progress: float, current_file: str | None = None) -> None:
            if on_progress:
                on_progress(step, progress, current_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            clone_url = repo_url
            if access_token:
                clone_url = clone_url.replace(
                    "https://", f"https://x-access-token:{access_token}@"
                )

            try:
                emit("cloning", 0.05)
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

            head_sha = subprocess.run(
                ["git", "-C", temp_dir, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            default_branch = subprocess.run(
                ["git", "-C", temp_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            emit("extracting_git", 0.15)
            git_stats = get_git_stats(temp_dir)
            all_entropies = [stats.get("ent", 0.0) for stats in git_stats.values()]
            all_ndev = [stats.get("ndev", 0) for stats in git_stats.values()]
            all_age = [stats.get("age_days", 0.0) for stats in git_stats.values()]

            emit("scanning_complexity", 0.40)
            # C2: parallel complexity scan — joblib uses all cores via loky
            source_files = get_source_files(temp_dir)
            rel_paths = [os.path.relpath(fp, temp_dir) for fp in source_files]
            complexity_list = Parallel(n_jobs=-1, backend="loky")(
                delayed(compute_complexity)(fp) for fp in source_files
            )
            raw_complexity = dict(zip(rel_paths, complexity_list))

            all_ccn = [c["avg_ccn"] for c in raw_complexity.values()]
            all_nloc = [c["avg_nloc"] for c in raw_complexity.values()]

            n_files = max(len(source_files), 1)

            emit("scoring_files", 0.55)

            # C1 Phase 1: collect per-file metadata and flatten all commit messages
            # for a single batch call to DistilBERT.
            file_data: list[dict] = []
            all_valid_messages: list[str] = []

            for fp in source_files:
                rel = os.path.relpath(fp, temp_dir)
                stats = git_stats.get(rel, {})
                commits = stats.get("commits", [])
                n_commits = stats.get("nf", 0)
                low_confidence = n_commits < LOW_CONFIDENCE_COMMIT_THRESHOLD

                low_info = [c for c in commits if len(c["message"].split()) < MIN_COMMIT_WORDS]
                valid = [c for c in commits if len(c["message"].split()) >= MIN_COMMIT_WORDS]
                low_info_ratio = len(low_info) / len(commits) if commits else 0.0

                ccn_score = percentile_rank(raw_complexity[rel]["avg_ccn"], all_ccn)
                nloc_score = percentile_rank(raw_complexity[rel]["avg_nloc"], all_nloc)
                complexity_score = (ccn_score * nloc_score) ** 0.5
                change_entropy = percentile_rank(stats.get("ent", 0.0), all_entropies)
                ndev_score = percentile_rank(stats.get("ndev", 0), all_ndev)
                age_score = percentile_rank(stats.get("age_days", 0.0), all_age)

                start_idx = len(all_valid_messages)
                all_valid_messages.extend(c["message"] for c in valid)
                end_idx = len(all_valid_messages)

                file_data.append({
                    "rel": rel,
                    "n_commits": n_commits,
                    "low_confidence": low_confidence,
                    "low_info_ratio": low_info_ratio,
                    "low_info_commits": low_info,
                    "valid_commits": valid,
                    "valid_slice": (start_idx, end_idx),
                    "complexity_score": complexity_score,
                    "change_entropy": change_entropy,
                    "ndev_score": ndev_score,
                    "age_score": age_score,
                    "avg_ccn": raw_complexity[rel]["avg_ccn"],
                    "avg_nloc": raw_complexity[rel]["avg_nloc"],
                })

            # C1 Phase 2: ONE batch call to DistilBERT across all valid commit messages.
            emit("scoring_files", 0.60)
            raw_sentiment_results = (
                ml_engine.sentiment_pipeline(all_valid_messages)
                if all_valid_messages
                else []
            )

            # C1 Phase 3: assemble per-file sentiment and build feature matrix
            # for high-confidence files.
            high_conf_indices: list[int] = []
            X_rows: list[list[float]] = []
            sentiment_scores: list[float] = []
            commit_sentiments_per_file: list[list[CommitSentiment]] = []

            for i, fd in enumerate(file_data):
                start, end = fd["valid_slice"]
                file_raw = raw_sentiment_results[start:end]
                valid_commits = fd["valid_commits"]

                # Build CommitSentiment list for this file
                sentiments: list[CommitSentiment] = []
                for c in fd["low_info_commits"]:
                    sentiments.append(CommitSentiment(
                        hash=c["hash"],
                        message=c["message"],
                        committed_at=c["committed_at"],
                        code=None,
                        low_info=True,
                        risk_probability=None,
                    ))

                risk_probs: list[float] = []
                for c, r in zip(valid_commits, file_raw):
                    best = max(r, key=lambda x: x["score"])
                    code = best["label"].lower()
                    risk_prob = min(
                        sum(x["score"] for x in r if x["label"].lower() == "caution"), 1.0
                    )
                    risk_probs.append(risk_prob)
                    sentiments.append(CommitSentiment(
                        hash=c["hash"],
                        message=c["message"],
                        committed_at=c["committed_at"],
                        code=code,
                        low_info=False,
                        risk_probability=risk_prob,
                    ))

                sentiment_score = sum(risk_probs) / len(risk_probs) if risk_probs else 0.0
                sentiment_scores.append(sentiment_score)
                commit_sentiments_per_file.append(sentiments)

                if not fd["low_confidence"]:
                    high_conf_indices.append(i)
                    X_rows.append([
                        sentiment_score,
                        fd["change_entropy"],
                        fd["ndev_score"],
                        fd["age_score"],
                        fd["complexity_score"],
                        fd["low_info_ratio"],
                        float(fd["n_commits"]),
                    ])

            # C1 Phase 4: ONE predict_proba call + ONE shap_values call.
            emit("scoring_files", 0.85)
            risk_score_map: dict[int, float] = {}
            shap_map: dict[int, SHAPBreakdown] = {}

            if X_rows:
                X_all = np.array(X_rows)
                X_scaled_all = ml_engine.scaler.transform(X_all)
                proba_all = ml_engine.risk_model.predict_proba(X_scaled_all)
                shap_vals_all = ml_engine.shap_explainer.shap_values(X_scaled_all)

                base_val = ml_engine.shap_explainer.expected_value
                base = float(base_val[1] if hasattr(base_val, "__len__") else base_val)

                for batch_idx, file_idx in enumerate(high_conf_indices):
                    risk_score_map[file_idx] = float(proba_all[batch_idx][1])
                    sv = (
                        shap_vals_all[1][batch_idx]
                        if isinstance(shap_vals_all, list)
                        else shap_vals_all[batch_idx]
                    )
                    shap_map[file_idx] = SHAPBreakdown(
                        base_rate=base,
                        sentiment_contrib=float(sv[0]),
                        entropy_contrib=float(sv[1]),
                        ndev_contrib=float(sv[2]),
                        age_contrib=float(sv[3]),
                        complexity_contrib=float(sv[4]),
                        low_info_contrib=float(sv[5]),
                        commits_contrib=float(sv[6]),
                    )

            # Assemble final results
            emit("scoring_files", 0.95)
            results: list[FileRiskResult] = []
            for i, fd in enumerate(file_data):
                results.append(FileRiskResult(
                    filename=fd["rel"],
                    risk_score=risk_score_map.get(i),
                    sentiment_score=sentiment_scores[i],
                    complexity_score=fd["complexity_score"],
                    change_entropy=fd["change_entropy"],
                    ndev_score=fd["ndev_score"],
                    age_score=fd["age_score"],
                    low_info_ratio=fd["low_info_ratio"],
                    commit_count=fd["n_commits"],
                    avg_ccn=fd["avg_ccn"],
                    avg_nloc=fd["avg_nloc"],
                    low_confidence=fd["low_confidence"],
                    shap_breakdown=shap_map.get(i),
                    commit_sentiments=commit_sentiments_per_file[i],
                ))

            results.sort(
                key=lambda x: x.risk_score if x.risk_score is not None else -1,
                reverse=True,
            )
            return AnalysisResult(
                file_results=results,
                threshold=ml_engine.threshold,
                head_sha=head_sha,
                default_branch=default_branch,
            )
