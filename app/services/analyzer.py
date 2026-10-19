import math
import shutil
import tempfile
import subprocess
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed

from ..schemas.analyze import FileRiskResult, SHAPBreakdown, CommitSentiment
from ..ml.git_extractor import get_git_stats
from ..ml.lizard_scanner import get_source_files, compute_complexity
from ..ml.helpers import (
    percentile_rank_presorted, sort_for_percentile, strip_prefix, is_low_info,
)
from ..ml.models import MLEngine
from ..ml.constants import (
    BERT_BATCH_SIZE, MAX_COMMITS_PER_FILE, DEFAULT_WINDOW_DAYS,
)

import logging

ml_engine = MLEngine()

logger = logging.getLogger(__name__)
# uvicorn's logger has a guaranteed handler — use it for timing output
_timing_log = logging.getLogger("uvicorn")


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

            t_total = time.perf_counter()
            # Strip token from URL before logging
            safe_url = repo_url.split("@")[-1] if "@" in repo_url else repo_url
            _timing_log.info(f"[timing] === analyzing: {safe_url} ===")

            try:
                emit("cloning", 0.05)
                logger.info(f"Cloning repository into {temp_dir}...")
                clone_cmd = [
                    "git",
                    "clone",
                    f"--shallow-since={DEFAULT_WINDOW_DAYS} days ago",
                    "--single-branch",
                ]
                if branch:
                    clone_cmd.extend(["--branch", branch])
                clone_cmd.extend([clone_url, temp_dir])

                t0 = time.perf_counter()
                try:
                    subprocess.run(clone_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    stderr = e.stderr.decode()
                    if "shallow info" in stderr:
                        # git bug: --shallow-since fails on some repos; retry full clone
                        logger.warning("Shallow clone failed (shallow info bug), retrying full clone...")
                        for entry in os.scandir(temp_dir):
                            if entry.is_dir(follow_symlinks=False):
                                shutil.rmtree(entry.path)
                            else:
                                os.remove(entry.path)
                        fallback_cmd = ["git", "clone", "--single-branch"]
                        if branch:
                            fallback_cmd.extend(["--branch", branch])
                        fallback_cmd.extend([clone_url, temp_dir])
                        subprocess.run(fallback_cmd, check=True, capture_output=True)
                    else:
                        raise ValueError(f"Git Clone Failed: {stderr}")
                _timing_log.info(f"[timing] clone: {time.perf_counter() - t0:.2f}s")
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
            t0 = time.perf_counter()
            git_stats = get_git_stats(temp_dir)
            _timing_log.info(f"[timing] git log extraction: {time.perf_counter() - t0:.2f}s")

            # Commit-level populations (unique commits in the window) for percentiles.
            # Ent/ndev/age are per-commit Kamei metrics, same for every file the commit touches.
            window_commits: dict[str, dict] = {}
            for stats in git_stats.values():
                for c in stats["commits"]:
                    window_commits.setdefault(c["hash"], c)
            sorted_entropy = sort_for_percentile([c["ent"] for c in window_commits.values()])
            sorted_ndev    = sort_for_percentile([float(c["ndev"]) for c in window_commits.values()])
            sorted_age     = sort_for_percentile([c["age_days"] for c in window_commits.values()])
            n_window_commits = len(window_commits)

            emit("scanning_complexity", 0.40)
            # C2: parallel complexity scan — joblib uses all cores via loky
            source_files = get_source_files(temp_dir)
            rel_paths = [os.path.relpath(fp, temp_dir) for fp in source_files]
            t0 = time.perf_counter()
            complexity_list = Parallel(n_jobs=-1, backend="loky")(
                delayed(compute_complexity)(fp) for fp in source_files
            )
            raw_complexity = dict(zip(rel_paths, complexity_list))
            _timing_log.info(
                f"[timing] complexity scan ({len(source_files)} files, parallel): "
                f"{time.perf_counter() - t0:.2f}s"
            )

            all_ccn = [c["avg_ccn"] for c in raw_complexity.values()]
            all_nloc = [c["avg_nloc"] for c in raw_complexity.values()]

            n_files = max(len(source_files), 1)

            # Pre-sort each metric array once — avoids O(n² log n) repeated sorts
            # inside the per-file loop (percentile_rank_presorted is O(log n) per call).
            sorted_ccn      = sort_for_percentile(all_ccn)
            sorted_nloc     = sort_for_percentile(all_nloc)
            n_metric        = len(sorted_ccn)

            emit("scoring_files", 0.55)

            # C1 Phase 1: collect per-file metadata and flatten all commit messages
            # for a single batch call to DistilBERT.
            t0 = time.perf_counter()
            file_data: list[dict] = []
            to_classify: dict[str, str] = {}  # hash -> stripped message (each commit scored once)

            for i, fp in enumerate(source_files):
                rel = os.path.relpath(fp, temp_dir)
                # Emit per-file so the loading screen shows current filename (0.55 → 0.65).
                emit("scoring_files", 0.55 + 0.10 * (i + 1) / n_files, rel)

                stats = git_stats.get(rel, {})
                all_commits = stats.get("commits", [])
                n_commits = stats.get("nf", 0)
                low_confidence = n_commits < ml_engine.min_commits

                # Features use every window commit, as in training. Each commit is classified
                # once (deduplicated by hash), so the full history costs no extra inference.
                low_info = [c for c in all_commits if is_low_info(c["message"])]
                valid = [c for c in all_commits if not is_low_info(c["message"])]
                low_info_ratio = len(low_info) / len(all_commits) if all_commits else 0.0

                def mean_pct(key, sorted_vals):
                    if not all_commits:
                        return 0.0
                    return sum(
                        percentile_rank_presorted(float(c[key]), sorted_vals, n_window_commits)
                        for c in all_commits
                    ) / len(all_commits)

                ccn_score = percentile_rank_presorted(raw_complexity[rel]["avg_ccn"], sorted_ccn, n_metric)
                nloc_score = percentile_rank_presorted(raw_complexity[rel]["avg_nloc"], sorted_nloc, n_metric)
                complexity_score = (ccn_score * nloc_score) ** 0.5
                change_entropy = mean_pct("ent", sorted_entropy)
                ndev_score = mean_pct("ndev", sorted_ndev)
                age_score = mean_pct("age_days", sorted_age)

                for c in valid:
                    to_classify.setdefault(c["hash"], strip_prefix(c["message"]))

                file_data.append({
                    "rel": rel,
                    "n_commits": n_commits,
                    "low_confidence": low_confidence,
                    "low_info_ratio": low_info_ratio,
                    "low_info_commits": low_info,
                    "valid_commits": valid,
                    # Payload only: the newest MAX_COMMITS_PER_FILE commits (newest-first order).
                    "shown": {c["hash"] for c in all_commits[:MAX_COMMITS_PER_FILE]},
                    "complexity_score": complexity_score,
                    "change_entropy": change_entropy,
                    "ndev_score": ndev_score,
                    "age_score": age_score,
                    "log_commits": math.log1p(n_commits),
                    "avg_ccn": raw_complexity[rel]["avg_ccn"],
                    "avg_nloc": raw_complexity[rel]["avg_nloc"],
                })

            _timing_log.info(
                f"[timing] data collection ({n_files} files, "
                f"{len(to_classify)} unique valid commit messages): "
                f"{time.perf_counter() - t0:.2f}s"
            )

            # C1 Phase 2: ONE batch call to DistilBERT across all valid commit messages.
            emit("scoring_files", 0.65)
            t0 = time.perf_counter()
            hashes = list(to_classify)
            raw_sentiment_results = (
                ml_engine.sentiment_pipeline(
                    [to_classify[h] or "empty" for h in hashes], batch_size=BERT_BATCH_SIZE
                )
                if hashes
                else []
            )
            risk_by_hash = {
                h: min(sum(x["score"] for x in r if x["label"].lower() == "caution"), 1.0)
                for h, r in zip(hashes, raw_sentiment_results)
            }
            code_by_hash = {
                h: max(r, key=lambda x: x["score"])["label"].lower()
                for h, r in zip(hashes, raw_sentiment_results)
            }
            # Training imputes a file with no valid commits with the project mean (0.5 if none).
            repo_mean_sentiment = (
                sum(risk_by_hash.values()) / len(risk_by_hash) if risk_by_hash else 0.5
            )
            _timing_log.info(
                f"[timing] DistilBERT batch inference ({len(hashes)} messages): "
                f"{time.perf_counter() - t0:.2f}s"
            )

            # C1 Phase 3: assemble per-file sentiment and build feature matrix
            # for high-confidence files.
            high_conf_indices: list[int] = []
            X_rows: list[list[float]] = []
            sentiment_scores: list[float] = []
            commit_sentiments_per_file: list[list[CommitSentiment]] = []

            for i, fd in enumerate(file_data):
                valid_commits = fd["valid_commits"]

                shown = fd["shown"]
                sentiments: list[CommitSentiment] = []
                for c in fd["low_info_commits"]:
                    if c["hash"] not in shown:
                        continue
                    sentiments.append(CommitSentiment(
                        hash=c["hash"],
                        message=c["message"],
                        committed_at=c["committed_at"],
                        code=None,
                        low_info=True,
                        risk_probability=None,
                        author=c.get("author"),
                        lines_added=c.get("lines_added"),
                        lines_deleted=c.get("lines_deleted"),
                        co_changed_files=c.get("co_changed_files"),
                    ))

                risk_probs: list[float] = []
                for c in valid_commits:
                    risk_prob = risk_by_hash[c["hash"]]
                    risk_probs.append(risk_prob)
                    if c["hash"] not in shown:
                        continue
                    sentiments.append(CommitSentiment(
                        hash=c["hash"],
                        message=c["message"],
                        committed_at=c["committed_at"],
                        code=code_by_hash[c["hash"]],
                        low_info=False,
                        risk_probability=risk_prob,
                        author=c.get("author"),
                        lines_added=c.get("lines_added"),
                        lines_deleted=c.get("lines_deleted"),
                        co_changed_files=c.get("co_changed_files"),
                    ))

                sentiment_score = sum(risk_probs) / len(risk_probs) if risk_probs else repo_mean_sentiment
                sentiment_scores.append(sentiment_score)
                commit_sentiments_per_file.append(sentiments)

                if not fd["low_confidence"]:
                    high_conf_indices.append(i)
                    feats = {**fd, "sentiment_score": sentiment_score}
                    X_rows.append([feats[f] for f in ml_engine.features])

            # C1 Phase 4: ONE predict_proba call + ONE shap_values call.
            emit("scoring_files", 0.85)
            t0 = time.perf_counter()
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
                    c = dict(zip(ml_engine.features, map(float, sv)))
                    shap_map[file_idx] = SHAPBreakdown(
                        base_rate=base,
                        sentiment_contrib=c.get("sentiment_score", 0.0),
                        entropy_contrib=c.get("change_entropy", 0.0),
                        ndev_contrib=c.get("ndev_score", 0.0),
                        age_contrib=c.get("age_score", 0.0),
                        complexity_contrib=c.get("complexity_score", 0.0),
                        low_info_contrib=c.get("low_info_ratio", 0.0),
                        commits_contrib=c.get("log_commits", 0.0),
                    )

            logger.info(
                f"[timing] risk model + SHAP ({len(X_rows)} high-conf files): "
                f"{time.perf_counter() - t0:.2f}s"
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
            _timing_log.info(f"[timing] total analysis: {time.perf_counter() - t_total:.2f}s")
            return AnalysisResult(
                file_results=results,
                threshold=ml_engine.threshold,
                head_sha=head_sha,
                default_branch=default_branch,
            )
