import math
import subprocess
from typing import Any
from .constants import DEFAULT_WINDOW_DAYS, SECONDS_PER_DAY


def _commit_entropy(churns: list[int]) -> float:
    """Kamei et al. (2013) change entropy: Shannon entropy of churn share across the
    files in one commit, normalised by log2(number of files)."""
    total = sum(churns)
    if len(churns) < 2 or total <= 0:
        return 0.0
    ent = -sum((c / total) * math.log2(c / total) for c in churns if c > 0)
    return max(0.0, ent / math.log2(len(churns)))


def get_git_stats(repo_path: str, window_days: int = DEFAULT_WINDOW_DAYS) -> dict[str, Any]:
    """
    Runs git log over the window and returns filepath -> {nf, commits}.
    Each commit entry carries per-commit Kamei metrics (ent, ndev, age_days),
    identical for every file the commit touches. Commits are newest-first.
    No message dedup: training never dedups either.
    """

    cmd = [
        "git",
        "-C",
        repo_path,
        "log",
        f"--since={window_days} days ago",
        "--format=%x1eCOMMIT\t%H\t%ae\t%at\t%s",
        "--numstat",
        "--no-renames",
        "--diff-filter=AM",
        "--",
        ".",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", check=True
        )
    except FileNotFoundError:
        raise RuntimeError("Git is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Git log command failed: {e.stderr}")

    # Parse: one entry per commit with its touched files (git log order = newest first).
    parsed: list[dict[str, Any]] = []
    for chunk in proc.stdout.split("\x1e"):
        lines = chunk.splitlines()
        if not lines or not lines[0].startswith("COMMIT\t"):
            continue

        header_parts = lines[0].split("\t", 4)
        commit = {
            "hash": header_parts[1] if len(header_parts) > 1 else "",
            "author": header_parts[2] if len(header_parts) > 2 else "",
            "timestamp": int(header_parts[3])
            if len(header_parts) > 3 and header_parts[3].isdigit()
            else 0,
            "message": header_parts[4] if len(header_parts) > 4 else "",
            "files": {},  # filepath -> (added, deleted)
        }
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                try:
                    la = int(parts[0]) if parts[0] not in ("-", "") else 0
                    ld = int(parts[1]) if parts[1] not in ("-", "") else 0
                except ValueError:
                    continue
                commit["files"][parts[2].strip()] = (la, ld)
        if commit["files"]:
            parsed.append(commit)

    # Per-commit Kamei metrics, walking oldest -> newest so "prior" history is known.
    file_authors: dict[str, set[str]] = {}
    file_last_ts: dict[str, int] = {}
    for commit in sorted(parsed, key=lambda c: c["timestamp"]):
        files = commit["files"]
        commit["ent"] = _commit_entropy([a + d for a, d in files.values()])

        prior_authors: set[str] = set()
        gaps: list[float] = []
        for fp in files:
            prior_authors |= file_authors.get(fp, set())
            if fp in file_last_ts:
                gaps.append((commit["timestamp"] - file_last_ts[fp]) / SECONDS_PER_DAY)
        commit["ndev"] = len(prior_authors)
        commit["age_days"] = max(0.0, sum(gaps) / len(gaps)) if gaps else 0.0

        for fp in files:
            file_authors.setdefault(fp, set()).add(commit["author"])
            file_last_ts[fp] = commit["timestamp"]

    result: dict[str, Any] = {}
    for commit in parsed:
        n_files = len(commit["files"])
        for fp, (la, ld) in commit["files"].items():
            entry = result.setdefault(fp, {"nf": 0, "commits": []})
            entry["nf"] += 1
            entry["commits"].append(
                {
                    "hash": commit["hash"],
                    "message": commit["message"],
                    "committed_at": commit["timestamp"],
                    "author": commit["author"],
                    "lines_added": la,
                    "lines_deleted": ld,
                    "co_changed_files": n_files,
                    "ent": commit["ent"],
                    "ndev": commit["ndev"],
                    "age_days": commit["age_days"],
                }
            )

    return result
