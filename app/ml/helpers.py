import bisect

from .constants import PERCENTILE_PRECISION


def percentile_rank(value: float, all_values: list[float]) -> float:
    """CDF-style percentile rank: fraction of values <= value.

    Matches training notebook exactly (bisect_right on sorted list).
    No min==max guard — uniform distribution correctly returns 1.0 for all
    members (100% of values are <= the common value), which is what the
    training data saw and what the model expects.
    """
    if not all_values or len(all_values) <= 1:
        return 0.0
    sorted_values = sorted(all_values)
    return round(bisect.bisect_right(sorted_values, value) / len(sorted_values), PERCENTILE_PRECISION)
