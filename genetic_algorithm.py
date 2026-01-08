import random
import time
import pandas as pd
from tsp_problem import TSPProblem
from pathlib import Path

from operators import (
    initialize_population, rank_select, tournament_select,
    roulette_select, order_crossover, partially_mapped_crossover,
    edge_recombination_crossover, swap_mutation, inversion_mutation,
    scramble_mutation
)

RESULTS_FILE = Path("results/experiments.csv")

CROSSOVER_METHODS = {
    "Order Crossover (OX)": order_crossover,
    "Partially Mapped Crossover (PMX)": partially_mapped_crossover,
    "Edge Recombination (ERX)": edge_recombination_crossover
}

MUTATION_METHODS = {
    "Swap Mutation": swap_mutation,
    "Inversion Mutation": inversion_mutation,
    "Scramble Mutation": scramble_mutation
}

""" FITNESS / FUNKCJA CELU """

def fitness (tour: list[int], tsp: TSPProblem) -> int:
    return tsp.tour_length(tour)

def _fitness_cache (population: list[list[int]], tsp: TSPProblem) -> dict[int, int]:
    cache: dict[int, int] = {}
    for tour in population:
        cache[id(tour)] = fitness(tour, tsp)
    return cache

def save_experiment (problem_type: str, best_distance: float, total_time: float, generations: int,
                     population_size: int, mutation_prob: float, crossover_prob: float,
                     elitism_percent: float, selection_method: str, crossover_method: str,
                     mutation_method: str, initial_distance: float, history: list[float], optimal_distance: int | None) -> None:

    RESULTS_FILE.parent.mkdir(exist_ok=True, parents=True)

    improvement = (initial_distance - best_distance) / initial_distance * 100 if initial_distance is not None else 0

    row = {
        "timestamp": time.time(),
        "problem": problem_type,
        "best_distance": best_distance,
        "optimal_distance": optimal_distance,
        "gap_percent": (best_distance - optimal_distance) / optimal_distance * 100 if optimal_distance is not None else None,
        "improvement_percent": improvement,
        "total_time_s": total_time,
        "generations": generations,
        "population_size": population_size,
        "mutation_prob": mutation_prob,
        "crossover_prob": crossover_prob,
        "elitism_percent": elitism_percent,
        "selection_method": selection_method,
        "mutation_method": mutation_method,
        "crossover_method": crossover_method,
        "initial_distance": initial_distance,
        "convergence_gen": next ((i for i, d in enumerate(history) if d == best_distance), len(history))
    }

    df_new = pd.DataFrame([row])

    if RESULTS_FILE.exists():
        df_new.to_csv(RESULTS_FILE, mode='a', index=False, header=False)
    else:
        df_new.to_csv(RESULTS_FILE, mode='w', index=False, header=True)

"""
ALGORYTM GŁÓWNY
"""

def run_genetic_algorithm(
        tsp: TSPProblem,
        population_size: int,
        generations: int,
        mutation_prob: float,
        crossover_prob: float,
        elitism_count: int,
        selection_method: str,
        crossover_method: str,
        mutation_method: str,
        rank_size: int,
        roulette_size: int,
        tournament_size: int,
        optimal_distance: int | None,
        problem_type: str,
        on_generation: callable = None, # callback dla UI
        generate_csv: bool = False,
        seed: int | None = None,
) -> tuple[list[int] | None, float, list[float], float]:
    start_time = time.time()
    if seed is None:
        random.seed()
    else:
        random.seed(seed)

    cities = list(tsp.coordinates.keys())
    population = initialize_population(cities, population_size)

    best_tour: list[int] = []
    best_distance = float('inf')
    initial_distance: float | None = None

    crossover_func = CROSSOVER_METHODS[crossover_method]
    mutation_func = MUTATION_METHODS[mutation_method]
    history: list[float] = []
    for generation in range(generations):
        fitness_cache = _fitness_cache(population, tsp)
        # Ocena fitness
        population_sorted = sorted(population, key=lambda tour: fitness_cache[id(tour)])

        current_best_tour = population_sorted[0]
        current_best_distance = fitness_cache[id(current_best_tour)]

        if generation == 0:
            initial_distance = current_best_distance

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = current_best_tour[:]

        if generate_csv:
            history.append(best_distance)

        # Aktualizacja wizualizacji co 10 generacji lub na końcu
        if on_generation and (generation % 10 == 0 or generation == generations - 1):
            elapsed_time = time.time() - start_time

            on_generation(
                generation=generation,
                generations_total=generations,
                tour=current_best_tour,
                distance=current_best_distance,
                hist=history,
                elapsed_time=elapsed_time,
                initial_distance=initial_distance,
                opt_distance=optimal_distance
            )
        # Elityzm -> ci mają gwarantowane przejście do następnej generacji
        elites: list[list[int]] = population_sorted[:elitism_count]
        remaining_count = population_size - elitism_count

        # Selekcja
        if selection_method == "Rank Selection":
            parents = rank_select(population, fitness_cache, rank_size, remaining_count)
        elif selection_method == "Tournament Selection":
            parents = tournament_select(population, fitness_cache, tournament_size, remaining_count)
        else:  # Roulette
            parents = roulette_select(population, fitness_cache, roulette_size, remaining_count)

        random.shuffle(parents)
        # Krzyżowanie
        offspring: list[list[int]] = []

        if len(parents) % 2 == 1:
            last_parent = parents.pop()
            offspring.append(last_parent[:])

        for i in range(0, len(parents), 2):
            p1, p2 = parents [i], parents[i + 1]
            if random.random() < crossover_prob:
                child1, child2 = crossover_func(p1, p2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1[:], p2[:]])

        # Mutacja
        mutated = [mutation_func(tour, mutation_prob) for tour in offspring]

        if len(mutated) < remaining_count:
            while len(mutated) < remaining_count:
                mutated.append(random.choice(mutated)[:])

        mutated = mutated[:remaining_count]

        # Nowa populacja
        population = elites + mutated

    total_time = time.time() - start_time
    if generate_csv:
        save_experiment (problem_type, best_distance, total_time, generations, population_size,
                   mutation_prob, crossover_prob, elitism_count, selection_method,
                   crossover_method, mutation_method, initial_distance, history, optimal_distance)

    return best_tour, best_distance, history, total_time