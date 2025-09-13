#!/usr/bin/env python

"""
Soccer Lineup Generator

This script generates a soccer lineup schedule based on player attributes and constraints.
"""

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, LpMinimize

def generate_lineups(roster_file, num_halfs=2, lines_per_half=4):
    """_summary_

    Args:
        roster_file (_type_): _description_
        num_halfs (int, optional): _description_. Defaults to 2.
        lines_per_half (int, optional): _description_. Defaults to 4.
    """
    # 1. Data Loading and Preprocessing
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

    # 2. Model Formulation
    model = LpProblem("SoccerLineupOptimization", LpMinimize)

    plays_in_line = LpVariable.dicts("PlayerInLine", (players, lines, positions), 0, 1, 'Binary')
    lines_played = LpVariable.dicts("LinesPlayed", players, 0, total_lines, 'Integer')
    min_lines = LpVariable("MinLines", 0, total_lines, 'Integer')
    max_lines = LpVariable("MaxLines", 0, total_lines, 'Integer')

    # Objective Function: Minimize the difference between max and min lines played
    model += max_lines - min_lines, "FairnessObjective"

    # Constraints to define lines_played
    for p in players:
        model += lines_played[p] == lpSum(plays_in_line[p][l][pos] for l in lines for pos in positions)

    # Constraints to define min_lines and max_lines
    for p in players:
        model += lines_played[p] >= min_lines
        model += lines_played[p] <= max_lines


    # Constraints
    # Each player can only play one position per line
    for p in players:
        for l in lines:
            model += lpSum(plays_in_line[p][l][pos] for pos in positions) <= 1

    # Each position in each line must be filled by exactly one player
    for l in lines:
        for pos in positions:
            model += lpSum(plays_in_line[p][l][pos] for p in players) == 1

    # Only goalies can play in the G position
    for l in lines:
        for p in players:
            if p not in goalies:
                model += plays_in_line[p][l]['G'] == 0

    # Defense line quality constraint
    for l in lines:
        model += lpSum(defense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in defense_positions) >= 7

    # Offense line quality constraint
    for l in lines:
        model += lpSum(offense_quality[p] * plays_in_line[p][l][pos] for p in players for pos in offense_positions) >= 6

    # Total line quality constraint
    for l in lines:
        model += lpSum((defense_quality[p] + offense_quality[p] + striker_quality[p]) * plays_in_line[p][l][pos] for p in players for pos in positions) >= 17

    # Footedness constraints
    left_positions = ['LD', 'LC']
    right_positions = ['RD', 'RC']

    for l in lines:
        for p in players:
            if p in righties:
                model += lpSum(plays_in_line[p][l][pos] for pos in left_positions) == 0
            if p in lefties:
                model += lpSum(plays_in_line[p][l][pos] for pos in right_positions) == 0

    # Consistent goalie constraint
    for h in range(num_halfs):
        start_line = h * lines_per_half
        lines_in_half = range(start_line, (h + 1) * lines_per_half)
        first_line_in_half = lines_in_half[0]
        for l in lines_in_half[1:]:
            for p in players:
                model += plays_in_line[p][l]['G'] == plays_in_line[p][first_line_in_half]['G']


    # 3. Solving and Output
    model.solve()

    if LpStatus[model.status] == 'Optimal':
        print("Optimal lineup schedule found:")
        for l in lines:
            print(f"\nLine {l + 1}:")
            for pos in positions:
                for p in players:
                    if plays_in_line[p][l][pos].varValue == 1:
                        print(f"  {pos}: {p}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    roster = "roster.csv"
    generate_lineups(roster)
