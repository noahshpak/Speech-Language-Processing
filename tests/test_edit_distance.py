import pytest
from chaptertwo import levenshtein

def test_robust_to_types():
    with pytest.raises(ValueError):
        levenshtein.edit_distance("hello", 12314)
    with pytest.raises(ValueError):
        levenshtein.edit_distance([], 2.3)

def test_simple_inputs():
    assert levenshtein.edit_distance("", "") == 0
    assert levenshtein.edit_distance("", "1") == 1
    assert levenshtein.edit_distance("1", "") == 1

def test_complete():
    assert levenshtein.edit_distance("INTENTION", "EXECUTION") == 8