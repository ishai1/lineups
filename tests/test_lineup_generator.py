import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lineups.fairness_generator import generate_lineups as generate_lineups_fairness
from lineups.multi_objective_generator import generate_lineups_multi_objective

def test_generate_lineups_fairness():
    try:
        generate_lineups_fairness("tests/dummy_roster.csv", num_halfs=1, lines_per_half=1)
    except Exception as e:
        assert False, f"generate_lineups_fairness raised an exception: {e}"

def test_generate_lineups_multi_objective():
    try:
        generate_lineups_multi_objective("tests/dummy_roster.csv", num_halfs=1, lines_per_half=1)
    except Exception as e:
        assert False, f"generate_lineups_multi_objective raised an exception: {e}"