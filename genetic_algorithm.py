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

def order_crossover (parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    parent_size = len(parent1)
    assert len(parent2) == parent_size and len(set(parent1)) == parent_size and len(set(parent2)) == parent_size
    """
    PRZYKŁADOWE DZIAŁANIE:
    start: 8
    end: 10
    Rodzic 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Rodzic 2: [3, 7, 5, 1, 9, 0, 2, 8, 6, 4]
    child: [None, None, None, None, None, None, None, None, 8, 9]
    child: [None, None, None, None, None, None, None, None, 6, 4]
    Dziecko 1: [3, 7, 5, 1, 0, 2, 6, 4, 8, 9]
    Dziecko 2: [0, 1, 2, 3, 5, 7, 8, 9, 6, 4]
    """
    start = random.randint(0,parent_size-1)
    end = random.randint(start+1,parent_size)

    def make_a_child_ox(pA: list[int], pB:list[int]) -> list[int]:
        child = [None] * parent_size
        child[start:end] = pA[start:end]
        taken = set(child[start:end])

        # modulo "przewija" indeksy od początku listy
        index = end % parent_size
        for i in range (parent_size):
            gene = pB[(end + i) % parent_size]
            if gene in taken:
                continue
            while child[index] is not None:
                index = (index + 1) % parent_size
            child[index] = gene
            index = (index + 1) % parent_size
        return child
    
    child1 = make_a_child_ox(parent1, parent2)
    child2 = make_a_child_ox(parent2, parent1)

    return child1, child2

def partially_mapped_crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    parent_size = len(parent1)
    assert len(parent2) == parent_size and len(set(parent1)) == parent_size and len(set(parent2)) == parent_size

    """
    PRZYKŁADOWE DZIAŁANIE:
    start: 8
    end: 10
    Rodzic 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Rodzic 2: [3, 7, 5, 1, 9, 0, 2, 8, 6, 4]
    child: [None, None, None, None, None, None, None, None, 8, 9]
    child: [None, None, None, None, None, None, None, None, 6, 4]
    Dziecko 1: [3, 7, 5, 1, 4, 0, 2, 6, 8, 9]
    Dziecko 2: [0, 1, 2, 3, 9, 5, 8, 7, 6, 4]
    """

    start = random.randint(0,parent_size-1)
    end = random.randint(start+1,parent_size)

    def lets_make_a_baby_pmx(pA: list[int], pB:list[int]) -> list[int]:
        child = [None] * parent_size
        child[start:end] = pA[start:end]

        for i in range (start, end):
            gene_from_pB = pB[i]

            if gene_from_pB in child[start:end]:
                continue
            
            position = i
            while child[position] is not None:
                value_in_child = child[position]
                position = pB.index(value_in_child)

            child[position] = gene_from_pB

        for i in range (parent_size):
            if child[i] is None:
                child[i] = pB[i]

        return child

    child1 = lets_make_a_baby_pmx(parent1, parent2)
    child2 = lets_make_a_baby_pmx(parent2, parent1)
    return child1, child2

def edge_recombination_crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    parent_size = len(parent1)
    assert len(parent2) == parent_size and len(set(parent1)) == parent_size and len(set(parent2)) == parent_size

    def build_edge_table(pA: list[int], pB: list[int]) -> dict[int, set[int]]:
        # utwórz pusty zbio©y z kluczem każdeho miasta
        edge_table = {city: set() for city in pA}

        def find_neighbours (parent) -> dict[int, set[int]]:
            for i in range(parent_size):
                current = parent[i]
                prev_city = parent[(i-1) % parent_size]
                next_city = parent[(i+1) % parent_size]
                edge_table[current].add(prev_city)
                edge_table[current].add(next_city)
        find_neighbours(pA)
        find_neighbours(pB)

        return edge_table
    def give_me_a_child_erx(pA: list[int], pB: list[int]) -> list[int]:
        edge_table = build_edge_table (pA, pB)
        child: list[int] = []

        # bieremy randomowe miasto z piyrszygo fatra/matuli
        current = random.choice(pA)  
        while len(child) < parent_size:
            child.append(current)

            # wyciepujemy `current` z wszystkich list sąsiedzkich
            for city in edge_table:
                edge_table[city].discard(current)

            if not edge_table[current]:
                remaining = [c for c in pA if c not in child]
                if not remaining:
                    break
                current = random.choice (remaining)
                continue
            neighbours = edge_table[current]
            min_edges = min(len(edge_table[n]) for n in neighbours)
            candidates = [n for n in neighbours if len(edge_table[n]) == min_edges]

            current = random.choice(candidates)
        return child

    child1 = give_me_a_child_erx(parent1, parent2)
    child2 = give_me_a_child_erx(parent2, parent1)

    return child1, child2

""" MUTATION """

def swap_mutation (tour: list[int], mutation_propability: float = 0.1) -> list[int]:
    pick = None
    mutation_propability*=2
    for i in range(len(tour)):
        if random.random() < mutation_propability:
            if pick is None:
                pick = i
            else:
                tour[pick], tour[i] = tour [i], tour[pick]
                pick = None
    return tour

def swap_mutate_population (population: list[list[int]], mutation_propability: float = 0.1
                       ) -> list[list[int]]:
    return [swap_mutation(tour, mutation_propability) for tour in population]

def genetic_algorithm (
        tsp: TSPProblem,
        population_size: int = 100,
        generations: int = 500,
        rank_size: int = 30,
        roulette_selection_size: int = 80,
        cross_propability: float = 0.5,
        mutation_propability: float = 0.02,
        elitism_count: float = 0.2,
        tournament_size: int = 3
        ):
    cities = list(tsp.coordinates.keys())
    population = initialize_population (cities=cities, population_size=population_size)

problem = TSPProblem("data/att48.tsp")
genetic_algorithm (problem, population_size=100, generations=500, rank_size=30,
                  roulette_selection_size = 80, cross_propability=0.8,
                  mutation_propability=0.1, elitism_count=0.2, tournament_size=3)