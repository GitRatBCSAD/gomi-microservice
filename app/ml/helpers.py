def percentile_rank(value: float, all_values: list[float]) -> float:
    if not all_values or len(all_values) <= 1:
        return 0.0
    if min(all_values) == max(all_values):
        return 0.0
    return round(sum(1 for x in all_values if x <= value) / len(all_values), 4)
