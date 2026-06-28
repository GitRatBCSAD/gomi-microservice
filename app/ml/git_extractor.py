import math
import subprocess
from typing import Any


def get_git_stats(repo_path: str, window_days: int = 180) -> dict[str, Any]:
    """
    Runs git log over the last 6 months and computes file-level statistics.
    It returns dict mapping filepath -> {ndev, age_days, ent, nf, messages}
    """

    cmd = [
    "git", "-C", repo_path, "log",
    f"--since={window_days} days ago",
    "--format=%x1eCOMMIT\t%H\t%ae\t%at\t%s",
    "--numstat", "--diff-filter=AM", "--", "."
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=True)
    except FileNotFoundError:
        raise RuntimeError("Git is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Git log command failed: {e.stderr}")

    raw_file_history = {}

    # Parsing the chunks
    for chunk in proc.stdout.split("\x1e"):
        lines = chunk.splitlines()
        if not lines or not lines[0].startswith("COMMIT\t"):
            continue

        header_parts = lines[0].split("\t", 4)
        commit_info = {
                "author": header_parts[2] if len(header_parts) > 2 else "",
                "timestamp": int(header_parts[3]) if len(header_parts) > 3 and header_parts[3].isdigit() else 0,
                "message": header_parts[4] if len(header_parts) > 4 else ""
                }
        
        # Git numstat liens

        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                try: 
                    la = int(parts[0]) if parts[0] not in ("-", "") else 0
                    ld = int(parts[1]) if parts[1] not in ("-", "") else 0
                    filepath = parts[2].strip()
                    raw_file_history.setdefault(filepath, []).append({**commit_info, "lines_added": la, "lines_deleted": ld})
                except ValueError:
                    continue

    # Calculate (Entropy, Dev Count, etc. ) for each
    result = {}
    for filepath, changes in raw_file_history.items():
        total_churn = sum(change["lines_added"] + change["lines_deleted"] for change in changes)
        timestamps = sorted(change["timestamp"] for change in changes)

        ent = 0.0
        if total_churn > 0:
            probs = [(change["lines_added"] + change["lines_deleted"]) / total_churn for change in changes]
            ent = -sum(p * math.log2(p) for p in probs if p > 0)

        seen_msgs = set()
        messages = []
        for change in changes:
            if change["message"] not in seen_msgs:
                messages.append(change["message"])
                seen_msgs.add(change["message"])
        result[filepath] = {
            "ndev": len(set(change["author"] for change in changes)),
            "age_days": (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) > 1 else 0,
            "ent": max(0.0, ent),
            "nf": len(changes),
            "messages": messages
        }

    return result



