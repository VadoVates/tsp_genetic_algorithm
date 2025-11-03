"""
genetic_algorithm.py – pętla GA: inicjalizacja populacji,
selekcja, krzyżowanie, mutacja, elityzm,
ocena fitness (używa tsp_problem.tour_length).
Zwraca najlepszą trasę i historię postępu.
"""

import random
from tsp_problem import TSPProblem

""" CODING / DECODING """

def pack (x1: int, x2: int) -> int:
    return (x1 << 5 | x2)

def unpack (coordinates: int) -> tuple[int, int]:
    x2 = coordinates & 0b11111
    x1 = (coordinates >> 5) & 0b11111
    return x1, x2

""" TWORZENIE LOSOWYCH PERMUTACJI """
def create_random_permutation (tour: list[int]) -> list[int]:
    cities = tour[:] #kopia
    random.shuffle(cities)
    return cities



""" FITNESS / FUNKCJA CELU """
def fitness (tour: list[int]) -> int:
    return TSPProblem.tour_length(tour)

""" RANK SELECTION """
def rank_select (tour: list[int], tour_length: int) -> list[int]:
    return (sorted(tour, key=fitness) [:tour_length])

""" TOURNAMENT SELECTION """
def tournament_select (tour: list[int], tournament_size: int = 3) -> int:
    if not tour:
        raise ValueError ("tour is empty")
    if not (1 <= tournament_size <= len(tour)):
        raise ValueError("Invalid tournament size")

    contenders = random.sample(tour, tournament_size)
    winner = min (contenders, key=fitness)
    return winner

""" ROULETTE SELECTION """

def roulette_select (tour: list[int]) -> list[int]:
    fitness_list = [fitness(coordinates) for coordinates in tour]
    total = sum (fitness_list)

    if total == 0: # jeżeli wszystkie dają różnicę zero, wtedy wybierz losowo
        return random.choices (tour, k = len(tour))

    fitness_accumulated = []
    s = 0
    for fit in fitness_list:
        s+=fit
        fitness_accumulated.append(s)

    selected = []
    for _ in range (len(tour)):
        r = random.randint (1, total)
        for i, accumulation in enumerate (fitness_accumulated):
            if r <= accumulation:
                selected.append(tour[i])
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

""" DESCRIBE VALUES """

def print_binary (value: int) -> str:
    return format(value & 0x3FF, '010b')

def describe_population (tour: list[int]) -> str:
    lines = []
    s = 0
    for i, coordinates in enumerate(tour, 1):
        x1, x2 = unpack(coordinates)
        fc = fitness(coordinates)
        s += fc
        lines.append(
            f"{i:>2}. bits={print_binary(coordinates)} (x1,x2)=({x1:>2},{x2:>2})  f={fc:>3}")
    lines.append(f"sum g = {s}")
    return "\n".join(lines)

""" MAIN ALGORITHM """

def genetic_algorithm (
        tour: list[int],
        generations: int = 100,
        cross_propability: float = 0.5,
        mutation_propability: float = 0.02
    ):
    best_overall = min (tour, key=fitness)
    
    for gen in range(generations):
        print()