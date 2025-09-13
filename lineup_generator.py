#!/usr/bin/env python

"""
Soccer Lineup Generator - Main Entry Point

This script allows the user to choose between two different lineup generation methods:
1.  Fairness: Minimizes the difference in playing time between players.
2.  Multi-Objective: First optimizes for fairness, then for total quality.
"""

import argparse
from lineups.fairness_generator import generate_lineups as generate_lineups_fairness
from lineups.multi_objective_generator import generate_lineups_multi_objective

def main():
    parser = argparse.ArgumentParser(description="Soccer Lineup Generator")
    parser.add_argument(
        "--mode",
        choices=["fairness", "multi-objective"],
        default="fairness",
        help="The optimization mode to use.",
    )
    parser.add_argument(
        "--roster",
        default="roster.csv",
        help="The path to the roster file.",
    )
    parser.add_argument(
        "--num_halfs",
        type=int,
        default=2,
        help="The number of halves to schedule.",
    )
    parser.add_argument(
        "--lines_per_half",
        type=int,
        default=4,
        help="The number of lines per half.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="The identifier for the run.",
    )
    args = parser.parse_args()

    if args.mode == "fairness":
        print("Running in fairness mode...")
        generate_lineups_fairness(args.roster, args.num_halfs, args.lines_per_half, args.run_id)
    elif args.mode == "multi-objective":
        print("Running in multi-objective mode...")
        generate_lineups_multi_objective(args.roster, args.num_halfs, args.lines_per_half, args.run_id)

if __name__ == "__main__":
    main()
