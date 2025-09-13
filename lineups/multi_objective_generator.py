#!/usr/bin/env python

"""
Soccer Lineup Generator - Multi-Objective Optimization

This script generates a soccer lineup schedule by first optimizing for fairness
(minimizing the difference in playing time) and then for total quality.
"""

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, LpMinimize, value

def generate_lineups_multi_objective(roster_file, num_halfs=2, lines_per_half=4):
    """
    Generates lineups using a two-step multi-objective optimization approach.
    """
    # --- Step 1: Optimize for Fairness ---

    # 1.1 Data Loading and Preprocessing
    positions = ['G', 'LD', 'CD', 'RD', 'S', 'LC', 'RC', 'FC']
    defense_positions = ['LD', 'CD', 'RD']
    offense_positions = ['LC', 'RC', 'FC']

    df = pd.read_csv(roster_file)
    quality_map = {'H': 3, 'M': 2, 'L': 1}
    df['Defense Quality'] = df['Defense Quality'].map(quality_map)
    df['Offense Quality'] = df['Offense Quality'].map(quality_map)
    df['Striker Quality'] = df['Striker Quality'].map(quality_map)

    lefties = df[df['Footedness'] == 'L']['Player'].tolist()
    righties = df[df['Footedness'] == 'R']['Player'].tolist()

    players = df['Player'].tolist()
    goalies = df[df['Goalie'] == 'Y']['Player'].tolist()
    
    defense_quality = dict(zip(df['Player'], df['Defense Quality']))
    offense_quality = dict(zip(df['Player'], df['Offense Quality']))
    striker_quality = dict(zip(df['Player'], df['Striker Quality']))

    total_lines = num_halfs * lines_per_half
    lines = range(total_lines)

    # 1.2 Model Formulation (Fairness)
    model_fairness = LpProblem("SoccerLineupFairness", LpMinimize)

    plays_in_line = LpVariable.dicts("PlayerInLine", (players, lines, positions), 0, 1, 'Binary')
    lines_played = LpVariable.dicts("LinesPlayed", players, 0, total_lines, 'Integer')
    min_lines = LpVariable("MinLines", 0, total_lines, 'Integer')
    max_lines = LpVariable("MaxLines", 0, total_lines, 'Integer')

    model_fairness += max_lines - min_lines, "FairnessObjective"

    for p in players:
        model_fairness += lines_played[p] == lpSum(plays_in_line[p][l][pos] for l in lines for pos in positions)
    for p in players:
        model_fairness += lines_played[p] >= min_lines
        model_fairness += lines_played[p] <= max_lines

    # Add all the base constraints
    for p in players:
        for l in lines:
            model_fairness += lpSum(plays_in_line[p][l][pos] for pos in positions) <= 1
    for l in lines:
        for pos in positions:
            model_fairness += lpSum(plays_in_line[p][l][pos] for p in players) == 1
    for l in lines:
        for p in players:
            if p not in goalies:
                model_fairness += plays_in_line[p][l]['G'] == 0
    for l in lines:
        model_fairness += lpSum(defense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in defense_positions) >= 7
    for l in lines:
        model_fairness += lpSum(offense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in offense_positions) >= 6
    for l in lines:
        model_fairness += lpSum((defense_quality[p] + offense_quality[p] + striker_quality[p]) * plays_in_line[p][l][pos] for p in players for pos in positions) >= 17
    left_positions = ['LD', 'LC']
    right_positions = ['RD', 'RC']
    for l in lines:
        for p in players:
            if p in righties:
                model_fairness += lpSum(plays_in_line[p][l][pos] for pos in left_positions) == 0
            if p in lefties:
                model_fairness += lpSum(plays_in_line[p][l][pos] for pos in right_positions) == 0

    # Consistent goalie constraint
    for h in range(num_halfs):
        start_line = h * lines_per_half
        lines_in_half = range(start_line, (h + 1) * lines_per_half)
        first_line_in_half = lines_in_half[0]
        for l in lines_in_half[1:]:
            for p in players:
                model_fairness += plays_in_line[p][l]['G'] == plays_in_line[p][first_line_in_half]['G']

    # 1.3 Solve for Fairness
    model_fairness.solve()

    if LpStatus[model_fairness.status] != 'Optimal':
        print("Could not find an optimal solution for fairness.")
        return

    optimal_fairness_diff = value(max_lines) - value(min_lines)
    print(f"Optimal fairness difference found: {optimal_fairness_diff}")

    # --- Step 2: Optimize for Quality, Constrained by Fairness ---

    # 2.1 Model Formulation (Quality)
    model_quality = LpProblem("SoccerLineupQuality", LpMaximize)

    # We reuse the same variables, but the objective and one constraint will change
    model_quality.variables = model_fairness.variables

    total_quality = lpSum(df.set_index('Player').loc[p]['Defense Quality'] * plays_in_line[p][l][pos] for p in players for l in lines for pos in positions) + \
                    lpSum(df.set_index('Player').loc[p]['Offense Quality'] * plays_in_line[p][l][pos] for p in players for l in lines for pos in positions) + \
                    lpSum(df.set_index('Player').loc[p]['Striker Quality'] * plays_in_line[p][l][pos] for p in players for l in lines for pos in positions)
    model_quality += total_quality

    # Add the fairness constraint
    model_quality += max_lines - min_lines == optimal_fairness_diff

    # Add all the base constraints again
    for p in players:
        for l in lines:
            model_quality += lpSum(plays_in_line[p][l][pos] for pos in positions) <= 1
    for l in lines:
        for pos in positions:
            model_quality += lpSum(plays_in_line[p][l][pos] for p in players) == 1
    for l in lines:
        for p in players:
            if p not in goalies:
                model_quality += plays_in_line[p][l]['G'] == 0
    for l in lines:
        model_quality += lpSum(defense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in defense_positions) >= 7
    for l in lines:
        model_quality += lpSum(offense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in offense_positions) >= 6
    for l in lines:
        model_quality += lpSum((defense_quality[p] + offense_quality[p] + striker_quality[p]) * plays_in_line[p][l][pos] for p in players for pos in positions) >= 17
    for l in lines:
        for p in players:
            if p in righties:
                model_quality += lpSum(plays_in_line[p][l][pos] for pos in left_positions) == 0
            if p in lefties:
                model_quality += lpSum(plays_in_line[p][l][pos] for pos in right_positions) == 0
    for p in players:
        model_quality += lines_played[p] == lpSum(plays_in_line[p][l][pos] for l in lines for pos in positions)
    for p in players:
        model_quality += lines_played[p] >= min_lines
        model_quality += lines_played[p] <= max_lines

    # Consistent goalie constraint
    for h in range(num_halfs):
        start_line = h * lines_per_half
        lines_in_half = range(start_line, (h + 1) * lines_per_half)
        first_line_in_half = lines_in_half[0]
        for l in lines_in_half[1:]:
            for p in players:
                model_quality += plays_in_line[p][l]['G'] == plays_in_line[p][first_line_in_half]['G']

    # 2.2 Solve for Quality
    model_quality.solve()

    # 3. Output
    if LpStatus[model_quality.status] == 'Optimal':
        print("Optimal lineup schedule found (multi-objective):")
        for l in lines:
            print(f"\nLine {l + 1}:")
            for pos in positions:
                for p in players:
                    if plays_in_line[p][l][pos].varValue == 1:
                        print(f"  {pos}: {p}")
    else:
        print("No optimal solution found for the quality objective.")

if __name__ == "__main__":
    roster = "roster.csv"
    generate_lineups_multi_objective(roster)
