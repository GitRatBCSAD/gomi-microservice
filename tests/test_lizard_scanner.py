"""Unit tests for app/ml/lizard_scanner.py — compute_complexity()."""
import os
import tempfile
import textwrap
import pytest
from app.ml.lizard_scanner import compute_complexity

_ZERO_RESULT = {"avg_ccn": 0.0, "avg_nloc": 0.0, "function_cnt": 0, "avg_param": 0.0}


def test_compute_complexity_nonexistent_file():
    """Non-existent path → lizard parse fails → all-zeros dict returned."""
    result = compute_complexity("/does/not/exist/fake_file.py")
    assert result == _ZERO_RESULT


def test_compute_complexity_empty_file():
    """Empty source file has no functions → all-zeros dict returned."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = compute_complexity(path)
        assert result == _ZERO_RESULT
    finally:
        os.unlink(path)


def test_compute_complexity_file_with_no_functions():
    """File with only module-level code (no defs) → all-zeros dict returned."""
    source = textwrap.dedent("""\
        X = 1
        Y = X + 2
        print(Y)
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = compute_complexity(path)
        assert result == _ZERO_RESULT
    finally:
        os.unlink(path)


def test_compute_complexity_simple_function():
    """File with one simple function → avg_ccn >= 1, function_cnt == 1."""
    source = textwrap.dedent("""\
        def greet(name):
            return f"Hello, {name}"
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = compute_complexity(path)
        assert result["function_cnt"] == 1
        assert result["avg_ccn"] >= 1.0
        assert result["avg_nloc"] > 0.0
    finally:
        os.unlink(path)


def test_compute_complexity_branching_raises_ccn():
    """Function with multiple branches → avg_ccn > 1."""
    source = textwrap.dedent("""\
        def classify(x):
            if x > 100:
                return "high"
            elif x > 50:
                return "medium"
            elif x > 10:
                return "low"
            else:
                return "none"
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = compute_complexity(path)
        assert result["avg_ccn"] > 1.0
        assert result["function_cnt"] == 1
    finally:
        os.unlink(path)


def test_compute_complexity_multiple_functions_averages():
    """avg_ccn and avg_nloc are averages across all functions, not a sum."""
    source = textwrap.dedent("""\
        def simple():
            return 1

        def branchy(x):
            if x > 0:
                return x
            return -x
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = compute_complexity(path)
        assert result["function_cnt"] == 2
        # branchy has CCN=2, simple has CCN=1; average = 1.5
        assert result["avg_ccn"] == pytest.approx(1.5, abs=0.1)
    finally:
        os.unlink(path)
