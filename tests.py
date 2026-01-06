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


def run_batch_tests(
    datasets: list[str] | None = None,
    population_sizes: list[int] | None = None,
    generations_list: list[int] | None = None,
    mutation_probs: list[float] | None = None,
    crossover_probs: list[float] | None = None,
    elitism_percents: list[float] | None = None,
    selection_methods: list[str] | None = None,
    crossover_methods: list[str] | None = None,
    mutation_methods: list[str] | None = None,
    rank_sizes: list[int] | None = None,
    tournament_sizes: list[int] | None = None,
    roulette_sizes: list[int] | None = None,
    repeats: int = 3
) -> None:
    """Uruchamia testy dla wszystkich kombinacji parametrów"""

    # Domyślne wartości
    datasets = datasets or list(DATASETS.keys())
    population_sizes = population_sizes or [100]
    generations_list = generations_list or [500]
    mutation_probs = mutation_probs or [0.1]
    crossover_probs = crossover_probs or [0.8]
    elitism_percents = elitism_percents or [10]
    selection_methods = selection_methods or ["Rank Selection", "Tournament Selection", "Roulette Selection"]
    crossover_methods = crossover_methods or ["Order Crossover (OX)", "Partially Mapped Crossover (PMX)", "Edge Recombination (ERX)"]
    mutation_methods = mutation_methods or ["Swap Mutation", "Inversion Mutation", "Scramble Mutation"]
    rank_sizes = rank_sizes or [30]
    tournament_sizes = tournament_sizes or [3]
    roulette_sizes = roulette_sizes or [50]

    # Generuj wszystkie kombinacje
    combinations = list(itertools.product(
        datasets,
        population_sizes,
        generations_list,
        mutation_probs,
        crossover_probs,
        elitism_percents,
        selection_methods,
        crossover_methods,
        mutation_methods
    ))

    total_tests = len(combinations) * repeats
    print(f"Łącznie testów: {total_tests}")
    print(f"Kombinacji: {len(combinations)}, powtórzeń: {repeats}")
    print("-" * 60)

    test_num = 0
    for (dataset, pop_size, gens, mut_prob, cross_prob, elite_pct,
         sel_method, cross_method, mut_method) in combinations:

        # Załaduj problem i optimum
        tsp = TSPProblem(DATASETS[dataset])
        optimal = optimal_tour(SOLUTION_PATHS[dataset])
        optimal_distance = tsp.tour_length(optimal)

        elitism_count = int(pop_size * elite_pct / 100)
        rank_size = min(rank_sizes[0], pop_size)
        tournament_size = tournament_sizes[0]
        roulette_size = min(roulette_sizes[0], pop_size)

        for repeat in range(repeats):
            test_num += 1
            print(f"[{test_num}/{total_tests}] {dataset} | {sel_method[:4]}-{cross_method[:3]}-{mut_method[:4]} | "
                  f"pop={pop_size}, gen={gens} | repeat {repeat + 1}/{repeats}")

            best_tour, best_distance, history, total_time = run_genetic_algorithm(
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
                on_generation=None  # bez wizualizacji
            )

            gap_percent = (best_distance - optimal_distance) / optimal_distance * 100
            print(f"    -> distance={best_distance:.0f}, gap={gap_percent:.2f}%, time={total_time:.2f}s")

    print("-" * 60)
    print("Zakończono! Wyniki w: results/experiments.csv")


if __name__ == "__main__":
    run_batch_tests(
        datasets=["ATT48", "Berlin52"],
        population_sizes=[100, 200],
        generations_list=[500, 1000],
        elitism_percents=[5, 10],
        mutation_probs=[0.05, 0.1, 0.15],
        crossover_probs=[0.7, 0.8, 0.9],
        selection_methods=["Rank Selection", "Tournament Selection", "Roulette Selection"],
        crossover_methods=["Order Crossover (OX)", "Partially Mapped Crossover (PMX)", "Edge Recombination (ERX)"],
        mutation_methods=["Swap Mutation", "Inversion Mutation", "Scramble Mutation"],
        rank_sizes=[30],
        tournament_sizes=[3],
        roulette_sizes=[50],
        repeats=3
    )