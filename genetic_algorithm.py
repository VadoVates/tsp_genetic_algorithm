"""
genetic_algorithm.py – pętla GA: inicjalizacja populacji,
selekcja, krzyżowanie, mutacja, elityzm,
ocena fitness (używa tsp_problem.tour_length).
Zwraca najlepszą trasę i historię postępu.
"""

import random
from tsp_problem import TSPProblem

""" INICJALIZACJA POPULACJI """

def create_random_permutation (cities: list[int]) -> list[int]:
    tour = cities[:] #kopia
    random.shuffle(tour)
    return tour

def initialize_population (cities: list[int], population_size: int) -> list[list[int]]:
    return [create_random_permutation(cities) for _ in range (population_size)]

""" FITNESS / FUNKCJA CELU """

# im mniej tym lepiej
def fitness (tour: list[int], tsp: TSPProblem) -> int:
    return tsp.tour_length(tour)

# odwrócenie logiki -> im mniej kilometrów, tym wyższy wynik
def fitness_normalized (tour: list[int], tsp: TSPProblem, worst_score: int):
    return (worst_score - fitness(tour, tsp) + 1)

""" SELECTION """

def rank_select (population: list[list[int]], rank_size: int,
                 tsp: TSPProblem) -> list[list[int]]:
    return (sorted(population, key=lambda tour: fitness(tour, tsp))
            [:rank_size])

def tournament_select (population: list[list[int]], tsp: TSPProblem,
                       tournament_size: int = 3) -> list[int]:
    if not population:
        raise ValueError ("population is empty")
    if not (1 <= tournament_size <= len(population)):
        raise ValueError("Invalid tournament size")

    contenders = random.sample(population, tournament_size)
    shortest_tour = min (contenders, key=lambda tour: fitness(tour, tsp))
    return shortest_tour

def roulette_select (population: list[list[int]], tsp: TSPProblem,
                     roulette_selection_size: int) -> list[list[int]]:
    contenders = random.sample(population, roulette_selection_size)
    worst_score = max(fitness(tour, tsp) for tour in contenders)
    fitness_list = [fitness_normalized(tour, tsp, worst_score) for tour in contenders]

    fitness_accumulated = []
    total = 0
    for fit in fitness_list:
        total+=fit
        fitness_accumulated.append(total)

    selected = []
    for _ in range (len(contenders)):
        r = random.randint (1, total)
        for i, accumulation in enumerate (fitness_accumulated):
            if r <= accumulation:
                selected.append(contenders[i])
                break

    return selected

""" CROSSOVER SECTION """

def crossover_pair (a: int, b: int, cut: int) -> tuple [int, int]:
    """
        cut_left ∈ [1..9], licząc od LEWEJ. Np. cut=3: |xxx|xxxxxxx
    """
    left_bits = 10 - cut
    right_mask = (1 << left_bits) - 1
    left_mask = 0b11_1111_1111
    ###    11_1111_1111 ^ np. 00_0000_0111 = 11_1111_1000
    left_mask = left_mask ^ right_mask
    c1 = (a & left_mask) | (b & right_mask)
    c1 = c1 & 0b11_1111_1111 # dla pewności że wynik ma 10 bitów
    c2 = (b & left_mask) | (a & right_mask)
    c2 = c2 & 0b11_1111_1111 # dla pewności że wynik ma 10 bitów
    return c1, c2

def one_point_crossover (tour: list[int], cross_propability: float = 0.5
                         ) -> list[int]:
    sh = tour[:] # klonowanie
    random.shuffle(sh)
    offspring: list[int] = []
    for i in range(0,len(sh),2):
        a = sh[i]
        # w razie przekręcenia licznika, modulo sparuje ostatni z pierwszym
        b = sh[(i+1) % len(sh)]
        if random.random() < cross_propability:
            cut = random.randint(1,9)
            c1, c2 = crossover_pair (a, b, cut)
            offspring += [c1, c2]
        else:
            offspring += [a, b]
    return offspring 

""" MUTATION """

def mutate (coordinates: int, mutation_propability: float = 0.02) -> int:
    c = coordinates
    for i in range(10):
        if random.random() < mutation_propability:
            # podmiana 1 bitu przez XOR
            c = c ^ (1 << i)

    return c

def mutate_population (tour: list[int], mutation_propability: float = 0.02
                       ) -> list[int]:
    out = []
    for coordinates in tour:
        out.append(mutate (coordinates, mutation_propability))
    return out

""" MAIN ALGORITHM """

def genetic_algorithm (
        tsp: TSPProblem,
        population_size: int = 100,
        generations: int = 500,
        rank_size: int = 30,
        roulette_selection_size: int = 80,
        cross_propability: float = 0.5,
        mutation_propability: float = 0.02,
        elitism_count: int = 2,
        tournament_size: int = 3,
        verbose: bool = True):
    cities = list(tsp.coordinates.keys())
    population = initialize_population (cities=cities, population_size=population_size)

    # for _ in range (generations):
    #     if verbose:
    #         print("".format())


problem = TSPProblem("data/att48.tsp")
genetic_algorithm (problem, population_size=100, generations=500, rank_size=30,
                  roulette_selection_size = 80, cross_propability=0.8,
                  mutation_propability=0.1, elitism_count=2, tournament_size=3,
                  verbose=True)