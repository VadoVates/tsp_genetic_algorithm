import random
import time
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import tsp_problem
from tsp_problem import TSPProblem
from pathlib import Path
from visualization import plot_tour, plot_convergence
from operators import (
    initialize_population, rank_select, tournament_select,
    roulette_select, order_crossover, partially_mapped_crossover,
    edge_recombination_crossover, swap_mutation, inversion_mutation,
    scramble_mutation
)

""" FITNESS / FUNKCJA CELU """

# zwraca długość trasy (im krótsza, tym lepsza)
def fitness (tour: list[int], tsp: TSPProblem) -> int:
    return tsp.tour_length(tour)

RESULTS_FILE = Path("results/experiments.csv")

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

################################################################################

def run_genetic_algorithm(
        tsp: tsp_problem.TSPProblem,
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
        map_placeholder,
        chart_placeholder,
        metrics_placeholder,
        progress_bar,
        status_text,
        elitism_percent: float,
        optimal_distance: int | None,
        problem_type: str
):

    random.seed()

    cities = list(tsp.coordinates.keys())
    population = initialize_population(cities, population_size)

    history = []
    best_tour = None
    best_distance = float('inf')
    start_time = time.time()
    initial_distance = None

    # Metody krzyżowania
    crossover_methods = {
        "Order Crossover (OX)": order_crossover,
        "Partially Mapped Crossover (PMX)": partially_mapped_crossover,
        "Edge Recombination (ERX)": edge_recombination_crossover
    }

    # Metody mutacji
    mutation_methods = {
        "Swap Mutation": swap_mutation,
        "Inversion Mutation": inversion_mutation,
        "Scramble Mutation": scramble_mutation
    }

    crossover_func = crossover_methods[crossover_method]
    mutation_func = mutation_methods[mutation_method]

    for generation in range(generations):
        fitness_cache = _fitness_cache(population, tsp)

        # Ocena fitness
        population_sorted = sorted(population, key=lambda tour: fitness_cache[id(tour)])

        current_best_tour = population_sorted[0]
        current_best_distance = fitness(current_best_tour, tsp)

        if generation == 0:
            initial_distance = current_best_distance

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = current_best_tour[:]

        history.append(best_distance)

        # Aktualizacja wizualizacji co 10 generacji lub na końcu
        if generation % 10 == 0 or generation == generations - 1:
            assert best_tour is not None
            # Mapa trasy
            with map_placeholder.container():
                fig_map = plot_tour(tsp, best_tour, f"Najlepsza trasa (Gen {generation + 1})")
                st.pyplot(fig_map)
                plt.close(fig_map)

            # Wykres zbieżności
            with chart_placeholder.container():
                fig_chart = plot_convergence(history, "Zbieżność algorytmu")
                st.pyplot(fig_chart)
                plt.close(fig_chart)

            # Metryki
            elapsed_time = time.time() - start_time
            improvement = ((initial_distance - best_distance) / initial_distance * 100) if initial_distance else 0
            diff_from_optimal = best_distance - optimal_distance if optimal_distance else None

            with metrics_placeholder.container():
                column1, column2, column3 = st.columns(3)
                with column1:
                    st.metric("Najlepsza odległość", f"{best_distance:.2f}")
                    st.metric("Generacja", f"{generation + 1}/{generations}")
                with column2:
                    if optimal_distance and diff_from_optimal is not None:
                        st.metric("Różnica od optimum", f"{diff_from_optimal:.2f}",
                                  delta=f"{(diff_from_optimal / optimal_distance * 100):.2f}%", delta_color="inverse")
                    st.metric("Czas wykonania", f"{elapsed_time:.2f}s")
                with column3:
                    st.metric("Poprawa", f"{improvement:.2f}%", delta=f"{improvement:.1f}%")
                    if optimal_distance:
                        st.metric("Optimum", f"{optimal_distance}")

            # Progress bar
            progress = (generation + 1) / generations
            progress_bar.progress(progress)
            status_text.text(f"Generacja {generation + 1}/{generations} - Dystans: {best_distance:.2f}")

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
        offspring = []
        for i in range(0, len(parents) - 1, 2): # ta magia "-1" to w razie nieparzystości zbioru
            if random.random() < crossover_prob:
                child1, child2 = crossover_func(parents[i], parents[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i + 1]])

        # Mutacja
        mutated = [mutation_func(tour, mutation_prob) for tour in offspring] [:remaining_count]

        # Nowa populacja
        population = elites + mutated

    total_time = time.time() - start_time
    save_experiment(problem_type, best_distance, total_time, generations, population_size,
                    mutation_prob, crossover_prob, elitism_percent, selection_method,
                    crossover_method, mutation_method, initial_distance, history, optimal_distance)

    return best_tour, best_distance, history, total_time