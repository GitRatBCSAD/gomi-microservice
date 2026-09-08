"""Unit tests for app/ml/helpers.py — percentile_rank() and percentile_rank_presorted()."""
import pytest
from app.ml.helpers import percentile_rank, percentile_rank_presorted, sort_for_percentile
from app.ml.constants import PERCENTILE_PRECISION


def test_percentile_normal():
    # 3 of 5 values are <= 3.0 → 3/5 = 0.6
    result = percentile_rank(3.0, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(0.6, abs=10 ** -PERCENTILE_PRECISION)


def test_percentile_value_at_min():
    # Only 1 value <= 1.0 → 1/5 = 0.2
    result = percentile_rank(1.0, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(0.2, abs=10 ** -PERCENTILE_PRECISION)


def test_percentile_value_at_max():
    # All 5 values <= 5.0 → 5/5 = 1.0
    result = percentile_rank(5.0, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(1.0, abs=10 ** -PERCENTILE_PRECISION)


def test_percentile_all_same():
    # All values identical — by design returns 1.0 (matches training behavior).
    # 100% of values are <= the common value; the model was trained on this behavior.
    result = percentile_rank(2.0, [2.0, 2.0, 2.0])
    assert result == pytest.approx(1.0, abs=10 ** -PERCENTILE_PRECISION)


def test_percentile_empty():
    assert percentile_rank(1.0, []) == 0.0


def test_percentile_single_element():
    # len <= 1 guard → 0.0
    assert percentile_rank(5.0, [5.0]) == 0.0


def test_percentile_precision():
    # Result should be rounded to PERCENTILE_PRECISION decimal places
    result = percentile_rank(1.0, [1.0, 2.0, 3.0])
    assert result == round(result, PERCENTILE_PRECISION)


def test_percentile_unsorted_input():
    # Input need not be sorted — function sorts internally
    result = percentile_rank(3.0, [5.0, 1.0, 3.0, 2.0, 4.0])
    assert result == pytest.approx(0.6, abs=10 ** -PERCENTILE_PRECISION)


# ---------------------------------------------------------------------------
# percentile_rank_presorted — must match percentile_rank exactly
# ---------------------------------------------------------------------------

def test_presorted_matches_original():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    sorted_vals = sort_for_percentile(values)
    n = len(values)
    for v in values:
        assert percentile_rank_presorted(v, sorted_vals, n) == percentile_rank(v, values)


def test_presorted_empty():
    assert percentile_rank_presorted(1.0, [], 0) == 0.0


def test_presorted_single():
    assert percentile_rank_presorted(5.0, [5.0], 1) == 0.0


def test_presorted_all_same():
    sorted_vals = sort_for_percentile([2.0, 2.0, 2.0])
    assert percentile_rank_presorted(2.0, sorted_vals, 3) == pytest.approx(1.0)
