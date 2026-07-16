#!/usr/bin/env python3
"""Held-out sweep for second-order context in monotonic syllable assignment."""

from __future__ import annotations

import itertools
import json
import math
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import tune_syllable_matcher_dp as base


def local_score(targets, pulses, components, row, column):
    timing, evidence, unexpected = components[row][column]
    return 0.50 * timing + 1.50 * evidence - 0.10 * unexpected + 0.60 * float(pulses[column]["score"])


def assign_context(targets, pulses, components, parameters):
    duration_weight, rhythm_weight, stress_weight, word_weight = parameters
    target_skip = 0.25
    pulse_skip = 0.05
    max_step = 2
    duration_floor = 0.100

    # State keeps the current and previous assignments. Retaining the previous
    # edge lets the next expansion compare a two-gap rhythm, while remaining a
    # narrow monotonic beam rather than a wide transcript search.
    states = {}
    state_keys_by_cell = {}
    for row in range(len(targets)):
        for column in range(len(pulses)):
            score = local_score(targets, pulses, components, row, column)
            score -= target_skip * row + pulse_skip * column
            key = (row, column, -1, -1)
            states[key] = (score, None)
            state_keys_by_cell.setdefault((row, column), []).append(key)

    for row in range(len(targets)):
        for column in range(len(pulses)):
            current_keys = state_keys_by_cell.get((row, column), ())
            for key in current_keys:
                score, _ = states[key]
                _, _, previous_row, previous_column = key
                for next_row in range(row + 1, min(row + max_step + 1, len(targets))):
                    for next_column in range(column + 1, min(column + max_step + 1, len(pulses))):
                        expected_gap = targets[next_row]["prior"] - targets[row]["prior"]
                        observed_gap = float(pulses[next_column]["center"]) - float(pulses[column]["center"])
                        duration_error = min(abs(observed_gap - expected_gap) / max(duration_floor, expected_gap), 3.0)
                        transition_cost = duration_weight * duration_error

                        if previous_row >= 0 and previous_column >= 0:
                            previous_expected = targets[row]["prior"] - targets[previous_row]["prior"]
                            previous_observed = float(pulses[column]["center"]) - float(pulses[previous_column]["center"])
                            expected_ratio = math.log(max(expected_gap, 0.025) / max(previous_expected, 0.025))
                            observed_ratio = math.log(max(observed_gap, 0.025) / max(previous_observed, 0.025))
                            transition_cost += rhythm_weight * min(abs(expected_ratio - observed_ratio), 2.0)

                        expected_stress_delta = targets[next_row]["stress"] - targets[row]["stress"]
                        observed_strength_delta = float(pulses[next_column]["score"]) - float(pulses[column]["score"])
                        if expected_stress_delta > 0 and observed_strength_delta < -0.05:
                            transition_cost += stress_weight
                        elif expected_stress_delta < 0 and observed_strength_delta > 0.05:
                            transition_cost += stress_weight

                        crosses_word = targets[next_row]["word"] != targets[row]["word"]
                        skips_within_word = next_row > row + 1 and not crosses_word
                        if skips_within_word:
                            transition_cost += word_weight

                        next_score = (
                            score
                            + local_score(targets, pulses, components, next_row, next_column)
                            - target_skip * (next_row - row - 1)
                            - pulse_skip * (next_column - column - 1)
                            - transition_cost
                        )
                        next_key = (next_row, next_column, row, column)
                        if next_key not in states:
                            state_keys_by_cell.setdefault((next_row, next_column), []).append(next_key)
                        if next_key not in states or next_score > states[next_key][0]:
                            states[next_key] = (next_score, key)

    best_key = None
    best_score = -1.0e30
    for key, (score, _) in states.items():
        row, column, _, _ = key
        final = score - target_skip * (len(targets) - row - 1) - pulse_skip * (len(pulses) - column - 1)
        if final > best_score:
            best_score, best_key = final, key
    assignments = []
    while best_key is not None:
        row, column, _, _ = best_key
        assignments.append((column, row))
        best_key = states[best_key][1]
    return list(reversed(assignments))


def score(cases, parameters):
    targets_total = assignments_total = exact = within_one = 0
    for _, targets, pulses, components in cases:
        targets_total += len(targets)
        assignments = assign_context(targets, pulses, components, parameters)
        assignments_total += len(assignments)
        used = [False] * len(targets)
        for pulse_index, predicted in assignments:
            center = float(pulses[pulse_index]["center"])
            actual = None
            error_best = 0.100001
            for index, target in enumerate(targets):
                if used[index]:
                    continue
                error = max(target["gold_start"] - center, center - target["gold_end"], 0.0)
                if error < error_best:
                    actual, error_best = index, error
            if actual is None:
                continue
            used[actual] = True
            exact += predicted == actual
            within_one += abs(predicted - actual) <= 1
    return {
        "precision": exact / max(assignments_total, 1),
        "recall": exact / max(targets_total, 1),
        "f1": 2 * exact / max(assignments_total + targets_total, 1),
        "within_one": within_one / max(assignments_total, 1),
        "assignments": assignments_total,
        "targets": targets_total,
    }


def main():
    cases = base.load_cases()
    groups = sorted({base.group_for(name) for name, *_ in cases})
    heldout_groups = set(groups[::5])
    train = [case for case in cases if base.group_for(case[0]) not in heldout_groups]
    heldout = [case for case in cases if base.group_for(case[0]) in heldout_groups]
    trials = []
    for parameters in itertools.product(
        (0.40, 0.60, 0.80, 1.00),
        (0.00, 0.10, 0.20, 0.35),
        (0.00, 0.05, 0.10),
        (0.00, 0.10, 0.20),
    ):
        result = score(train, parameters)
        trials.append((result["f1"], min(result["precision"], result["recall"]), parameters, result))
    best = max(trials)
    report = {
        "cases": len(cases),
        "parameters": best[2],
        "train": best[3],
        "heldout": score(heldout, best[2]),
        "all": score(cases, best[2]),
        "ablations_heldout": {},
    }
    for name, index in (("rhythm", 1), ("stress", 2), ("word", 3)):
        ablated = list(best[2])
        ablated[index] = 0.0
        report["ablations_heldout"][name] = score(heldout, tuple(ablated))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
