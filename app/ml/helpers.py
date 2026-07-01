def percentile_rank(value: float, all_values: list[float]) -> float:
    """
    Returns the percentile rank of a value relative to all_values (between 0.0 and 1.0)
    For example, if a file has a higher CC score than 80% of all files, it would return 0.8
    """

    if not all_values or len(all_values) == 0:
        return 0.0

    percentile = sum(1 for x in all_values if x < value) / len(all_values)

    return round(percentile, 4)
