"""
tests.py - Batch testing dla algorytmu genetycznego TSP
Uruchom: python tests.py
"""

import itertools
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

def float_range (start:float, stop:float, step:float):
    result = []
    current = start
    while current <= stop + step/2:
        result.append(round(current,4))
        current += step
    return result

def int_range (start:int, stop:int, step:int):
    return list(range(start, stop + 1, step))

CONFIG = {
    # datasety
    "datasets": ["ATT48", "Berlin52"],

    # parametry populacji
    "population_size": int_range(50, 500, 10),
    "generations": int_range(100, 2000, 50),
    "elitism_percent": float_range(0, 0.2, 0.01),

    # prawdopodobieństwa
    "mutation_propablity": float_range(0.01, 0.2, 0.01),
    "crossover_propablity": float_range(0.5, 1.0, 0.05),

    # metody selekcji
    "selection_methods": [
        "Rank Selection",
        "Tournament Selection",
        "Roulette Selection"
    ],
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

    # parametry selekcji
    "rank_size": int_range(10, 50, 5),
    "tournament_size": int_range(2, 10, 1),
    "roulette_size": int_range(10, 50, 10),

    # powtórzeń na kombinację
    "repeats": 1
}

def calculate_total_tests(config:dict) -> int:
    count = (
            len(config["datasets"]) *
            len(config["population_size"]) *
            len(config["generations"]) *
            len(config["elitism_percent"]) *
            len(config["mutation_prob"]) *
            len(config["crossover_prob"]) *
            len(config["selection_methods"]) *
            len(config["crossover_methods"]) *
            len(config["mutation_methods"]) *
            config["repeats"]
    )
    return count

def run_batch_tests(config: dict) -> None:
    total_tests = calculate_total_tests(config)
    print(f"Łącznie testów: {total_tests}")
    print(f"Powtórzeń na kombinację: {config['repeats']}")
    print("-" * 60)

    combinations = list(itertools.product(
        config["datasets"],
        config["population_size"],
        config["generations"],
        config["elitism_percent"],
        config["mutation_prob"],
        config["crossover_prob"],
        config["selection_methods"],
        config["crossover_methods"],
        config["mutation_methods"]
    ))

    test_num = 0
    for (dataset, pop_size, gens, elite_pct, mut_prob, cross_prob,
         sel_method, cross_method, mut_method) in combinations:
        tsp = TSPProblem(DATASETS[dataset])
        optimal = optimal_tour(SOLUTION_PATHS[dataset])
        optimal_distance = tsp.tour_length(optimal)

        elitism_count = int(pop_size * elite_pct / 100)
        rank_size = min(config["rank_size"], pop_size)
        roulette_size = min(config["roulette_size"], pop_size)
        tournament_size = config["tournament_size"]

        for repeat in range(config["repeats"]):
            test_num += 1

            run_genetic_algorithm(
                tsp=tsp,
                population_size=pop_size,
                generations=gens,
                mutation_prob=mut_prob,
                crossover_prob=cross_prob,
                elitism_count=elitism_count,
                selection_method=sel_method,
                crossover_method=cross_method,
                mutation_method=mut_method,
                rank_size=rank_size,
                roulette_size=roulette_size,
                tournament_size=tournament_size,
                elitism_percent=elite_pct,
                optimal_distance=optimal_distance,
                problem_type=dataset,
                on_generation=None  # bez wizualizacji!
            )

if __name__ == "__main__":
    total = calculate_total_tests(CONFIG)
    print(f"Zaplanowano {total} testów")

    run_batch_tests(CONFIG)