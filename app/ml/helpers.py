import bisect
import re

from .constants import PERCENTILE_PRECISION

# --- shared with gomi-ml/gomi_train.ipynb: keep byte-identical ---------------
CONVENTIONAL_COMMIT_RE = r'^[a-z]+(\([^)]+\))?!?:\s*'
LOW_INFO_TOKEN_THRESHOLD = 5
# Kamei et al. (2013) fix keywords, word-bounded so "prefix" does not match "fix".
FIX_RE = re.compile(
    r'\b(bugs?|bugfix(es)?|fix(e[sd]|ing)?|wrong|errors?|fail(s|ed|ing|ures?)?|problems?|patch(es|ed)?)\b',
    re.IGNORECASE,
)
# Leading tracker ID ("Bug 424742 - ...", Eclipse style) is on every change, not only fixes.
BUG_ID_RE = re.compile(r'^\[?bug\s*:?\s*\d+\]?\s*[-:]?\s*', re.IGNORECASE)


def strip_prefix(msg):
    return re.sub(CONVENTIONAL_COMMIT_RE, '', msg or '', flags=re.IGNORECASE).strip()


def is_low_info(msg):
    return len(strip_prefix(msg).split()) < LOW_INFO_TOKEN_THRESHOLD


def is_fix(msg):
    return FIX_RE.search(BUG_ID_RE.sub('', strip_prefix(msg))) is not None
# ------------------------------------------------------------------------------


def percentile_rank(value: float, all_values: list[float]) -> float:
    """CDF-style percentile rank: fraction of values <= value.

    Matches training notebook exactly (bisect_right on sorted list).
    No min==max guard — uniform distribution correctly returns 1.0 for all
    members (100% of values are <= the common value), which is what the
    training data saw and what the model expects.

    NOTE: sorts all_values on every call — O(n log n) per call.
    Use percentile_rank_presorted() in hot loops where all_values is stable.
    """
    if not all_values or len(all_values) <= 1:
        return 0.0
    sorted_values = sorted(all_values)
    return round(bisect.bisect_right(sorted_values, value) / len(sorted_values), PERCENTILE_PRECISION)


def sort_for_percentile(all_values: list[float]) -> list[float]:
    """Pre-sort a metric array once before a loop. Pass result to percentile_rank_presorted().

    Pattern for hot loops (O(n log n) total instead of O(n² log n)):
        sorted_ccn = sort_for_percentile(all_ccn)
        for file in files:
            score = percentile_rank_presorted(file_ccn, sorted_ccn, len(all_ccn))
    """
    return sorted(all_values)


def percentile_rank_presorted(value: float, sorted_values: list[float], n: int) -> float:
    """CDF-style percentile rank using a pre-sorted list. O(log n) per call.

    Identical result to percentile_rank() — same bisect_right logic, same rounding.
    Requires sorted_values = sort_for_percentile(all_values) called once before the loop.
    """
    if n <= 1:
        return 0.0
    return round(bisect.bisect_right(sorted_values, value) / n, PERCENTILE_PRECISION)
