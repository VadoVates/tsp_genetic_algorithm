"""
tests.py - Batch testing dla algorytmu genetycznego TSP
Uruchom: python tests.py
"""

import itertools
import csv
import time
from pathlib import Path
from tsp_problem import TSPProblem, optimal_tour
from genetic_algorithm import run_genetic_algorithm

DATASETS = {
    "ATT48": "data/att48.tsp",
    "Berlin52": "data/berlin52.tsp"
}

SOLUTION_PATHS = {
    "ATT48": "data/att48.opt.tour",
    "Berlin52": "data/berlin52.opt.tour"
}

RESULTS_FILE = Path("results/experiments.csv")
FLUSH_EVERY = 500

def float_range(start: float, stop: float, step: float) -> list[float]:
    result = []
    current = start
    while current <= stop + step / 2:
        result.append(round(current, 4))
        current += step
    return result

def int_range(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop + 1, step))

COMMON_CONFIG = {
    # parametry populacji
    "population_size": int_range(50, 500, 50),
    "generations": int_range(100, 2000, 100),
    "elitism_percent": float_range(0, 20, 2),

    # prawdopodobieństwa
    "mutation_prob": float_range(0.01, 0.2, 0.02),
    "crossover_prob": float_range(0.5, 1.0, 0.1),

    "crossover_methods": [
        "Order Crossover (OX)",
        "Partially Mapped Crossover (PMX)",
        "Edge Recombination (ERX)"
    ],
    "mutation_methods": [
        "Swap Mutation",
        "Inversion Mutation",
        "Scramble Mutation"
    ],
    "repeats": 1
}

SELECTION_CONFIGS = {
    "Rank Selection": {
        "rank_size": int_range(10, 50, 10)
    },
    "Tournament Selection": {
        "tournament_size": int_range(2, 10, 2)
    },
    "Roulette Selection": {
        "roulette_size": int_range(10, 50, 10)
    }
}

OPERATOR_CONFIGS = {
    "Selection": [
        "Rank Selection",
        "Tournament Selection",
        "Roulette Selection"
    ],
    "Crossover": [
        "Order Crossover (OX)",
        "Partially Mapped Crossover (PMX)",
        "Edge Recombination (ERX)"
    ],
    "Mutation": [
        "Swap Mutation",
        "Inversion Mutation",
        "Scramble Mutation"
    ]
}

def calculate_tests_for_selection(common_config:dict, selection_param_values: list) -> int:
    return (
            len(common_config["population_size"]) *
            len(common_config["generations"]) *
            len(common_config["elitism_percent"]) *
            len(common_config["mutation_prob"]) *
            len(common_config["crossover_prob"]) *
            len(common_config["crossover_methods"]) *
            len(common_config["mutation_methods"]) *
            len(selection_param_values) *
            common_config["repeats"]
    )

def calculate_total_tests(common_config: dict, selection_methods: list[str]) -> int:
    total_tests = 0
    for sel_method in selection_methods:
        sel_config = SELECTION_CONFIGS[sel_method]
        param_values = list(sel_config.values())[0]
        total_tests += calculate_tests_for_selection(common_config, param_values)
    return total_tests

def _csv_append_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

best_result = {
    "distance": float('inf'),
    "gap_percent": float('inf'),
    "config": None
}

def run_tests_for_selection(
    dataset: str,
    selection_method: str,
    crossover_method: str,
    mutation_method: str,
    common_config: dict,
    test_num_start: int,
    total_tests: int,
    buffer_rows: list[dict],
    csv_fieldnames: list[str]
) -> int:
    global best_result

    tsp = TSPProblem(DATASETS[dataset])
    optimal = optimal_tour(SOLUTION_PATHS[dataset])
    optimal_distance = tsp.tour_length(optimal)

    sel_config = SELECTION_CONFIGS[selection_method]
    param_name = list(sel_config.keys())[0]
    param_values = sel_config[param_name]

    combinations = itertools.product(
        common_config["population_size"],
        common_config["generations"],
        common_config["elitism_percent"],
        common_config["mutation_prob"],
        common_config["crossover_prob"],
        param_values
    )
    test_num = test_num_start

    for (pop_size, gens, elite_pct, mut_prob, cross_prob, sel_param) in combinations:
        # selekcja: tylko właściwy parametr ma znaczenie
        if selection_method == "Rank Selection":
            rank_size = min(sel_param, pop_size)
            tournament_size = 3
            roulette_size = 50
        elif selection_method == "Tournament Selection":
            rank_size = 30
            tournament_size = min(sel_param, pop_size)
            roulette_size = 50
        else:  # Roulette Selection
            rank_size = 30
            tournament_size = 3
            roulette_size = min(sel_param, pop_size)

        elitism_count = int(pop_size * elite_pct / 100)

        for repeat in range(common_config["repeats"]):
            test_num += 1

            # zakładam, że dodałeś:
            # save: bool, keep_history: bool
            best_tour, best_distance, history, total_time = run_genetic_algorithm(
                tsp=tsp,
                population_size=pop_size,
                generations=gens,
                mutation_prob=mut_prob,
                crossover_prob=cross_prob,
                elitism_count=elitism_count,
                selection_method=selection_method,
                crossover_method=crossover_method,
                mutation_method=mutation_method,
                rank_size=rank_size,
                roulette_size=roulette_size,
                tournament_size=tournament_size,
                optimal_distance=optimal_distance,
                problem_type=dataset,
                on_generation=None,
                generate_csv = False
            )

            gap_percent = (best_distance - optimal_distance) / optimal_distance * 100

            if gap_percent < best_result["gap_percent"]:
                best_result = {
                    "distance": best_distance,
                    "gap_percent": gap_percent,
                    "config": {
                        "dataset": dataset,
                        "pop_size": pop_size,
                        "generations": gens,
                        "elite_pct": elite_pct,
                        "mut_prob": mut_prob,
                        "cross_prob": cross_prob,
                        "selection": selection_method,
                        "sel_param": sel_param,
                        "crossover": crossover_method,
                        "mutation": mutation_method
                    }
                }

            # bufor wyniku do CSV
            row = {
                "timestamp": time.time(),
                "problem": dataset,
                "best_distance": best_distance,
                "optimal_distance": optimal_distance,
                "gap_percent": gap_percent,
                "total_time_s": total_time,
                "generations": gens,
                "population_size": pop_size,
                "mutation_prob": mut_prob,
                "crossover_prob": cross_prob,
                "elitism_percent": elite_pct,
                "selection_method": selection_method,
                "selection_param_name": param_name,
                "selection_param_value": sel_param,
                "crossover_method": crossover_method,
                "mutation_method": mutation_method,
                "repeat": repeat
            }
            buffer_rows.append(row)

            # flush co 500 testów
            if len(buffer_rows) >= FLUSH_EVERY:
                _csv_append_rows(RESULTS_FILE, buffer_rows, csv_fieldnames)
                buffer_rows.clear()

            if test_num % 100 == 0:
                print(f"[{test_num}/{total_tests}] ({100 * test_num / total_tests:.1f}%)")

            if test_num % 1000 == 0:
                print(f"\n{'=' * 60}")
                print(f"TOP 1 po {test_num} testach:")
                print(f"  Gap: {best_result['gap_percent']:.2f}%")
                print(f"  Distance: {best_result['distance']:.0f}")
                print(f"  Config: {best_result['config']}")
                print(f"{'=' * 60}\n")

    return test_num


def run_batch_tests(
        dataset: str,
        selection_methods: list[str],
        crossover_methods: list[str],
        mutation_methods: list[str]
) -> None:
    global best_result

    best_result = {"distance": float("inf"), "gap_percent": float("inf"), "config": None}

    # CSV schema (stałe kolumny)
    csv_fieldnames = [
        "timestamp",
        "problem",
        "best_distance",
        "optimal_distance",
        "gap_percent",
        "total_time_s",
        "generations",
        "population_size",
        "mutation_prob",
        "crossover_prob",
        "elitism_percent",
        "selection_method",
        "selection_param_name",
        "selection_param_value",
        "crossover_method",
        "mutation_method",
        "repeat"
    ]

    # Bufor wyników do CSV
    buffer_rows: list[dict] = []

    # Kluczowa zmiana: iterujesz tylko po wybranych metodach (zamiast po wszystkich zawsze)
    # selection_methods/crossover_methods/mutation_methods mogą mieć długość 1 (to jest ten tryb)
    total_tests = 0
    for sel in selection_methods:
        sel_param_values = list(SELECTION_CONFIGS[sel].values())[0]
        total_tests += (
                len(COMMON_CONFIG["population_size"]) *
                len(COMMON_CONFIG["generations"]) *
                len(COMMON_CONFIG["elitism_percent"]) *
                len(COMMON_CONFIG["mutation_prob"]) *
                len(COMMON_CONFIG["crossover_prob"]) *
                len(sel_param_values) *
                COMMON_CONFIG["repeats"] *
                len(crossover_methods) *
                len(mutation_methods)
        )

    print(f"Dataset: {dataset}")
    print(f"Selection: {selection_methods}")
    print(f"Crossover: {crossover_methods}")
    print(f"Mutation: {mutation_methods}")
    print(f"Łącznie testów: {total_tests}")
    print("-" * 60)

    test_num = 0
    for sel_method in selection_methods:
        for cross_method in crossover_methods:
            for mut_method in mutation_methods:
                print(f"\n>>> {sel_method} | {cross_method} | {mut_method}")
                test_num = run_tests_for_selection(
                    dataset=dataset,
                    selection_method=sel_method,
                    crossover_method=cross_method,
                    mutation_method=mut_method,
                    common_config=COMMON_CONFIG,
                    test_num_start=test_num,
                    total_tests=total_tests,
                    buffer_rows=buffer_rows,
                    csv_fieldnames=csv_fieldnames
                )

    # flush końcowy
    if buffer_rows:
        _csv_append_rows(RESULTS_FILE, buffer_rows, csv_fieldnames)
        buffer_rows.clear()

    print("-" * 60)
    print("ZAKOŃCZONO!")
    print(f"\nNAJLEPSZY WYNIK:")
    print(f"  Gap: {best_result['gap_percent']:.2f}%")
    print(f"  Distance: {best_result['distance']:.0f}")
    print(f"  Config: {best_result['config']}")
    print(f"\nWyniki zapisane w: {RESULTS_FILE}")


if __name__ == "__main__":
    DATASET = "ATT48"  # lub "Berlin52"

    # WYBÓR: możesz zawęzić do jednej rzeczy w każdej kategorii (to jest to, o co prosisz)
    # np. tylko rank + tylko OX + tylko inversion:
    # selection_methods = ["Rank Selection"]
    # crossover_methods = ["Order Crossover (OX)"]
    # mutation_methods = ["Inversion Mutation"]

    selection_methods = [
        "Rank Selection",
        "Tournament Selection",
        "Roulette Selection"
    ]
    crossover_methods = COMMON_CONFIG["crossover_methods"]
    mutation_methods = COMMON_CONFIG["mutation_methods"]

    # szacunek czasu zostawiłem out, bo i tak zależy od realnego runtime; liczba testów jest tu źródłem prawdy
    # (i nadal może być kosmiczna, jeśli nie zawęzisz).
    total_tests = 0
    for sel in selection_methods:
        sel_param_values = list(SELECTION_CONFIGS[sel].values())[0]
        total_tests += (
                len(COMMON_CONFIG["population_size"]) *
                len(COMMON_CONFIG["generations"]) *
                len(COMMON_CONFIG["elitism_percent"]) *
                len(COMMON_CONFIG["mutation_prob"]) *
                len(COMMON_CONFIG["crossover_prob"]) *
                len(sel_param_values) *
                COMMON_CONFIG["repeats"] *
                len(crossover_methods) *
                len(mutation_methods)
        )

    print(f"Dataset: {DATASET}")
    print(f"Zaplanowano {total_tests} testów")

    confirm = input("Kontynuować? (t/n): ")
    if confirm.lower() == "t":
        run_batch_tests(DATASET, selection_methods, crossover_methods, mutation_methods)
    else:
        print("Anulowano.")
