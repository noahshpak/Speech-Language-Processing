import pytest
from chaptertwo import maxmatch

def test_incorrect_english_parse():
    input = 'wecanonlyseeashortdistanceahead'
    output = ['we', 'canon', 'l', 'y', 'see', 'ash', 'or', 't', 'distance', 'ah', 'ea', 'd']
    assert maxmatch.parse(input) == output