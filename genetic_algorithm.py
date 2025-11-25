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

# zwraca długość trasy (im krótsza, tym lepsza)
def fitness (tour: list[int], tsp: TSPProblem) -> int:
    return tsp.tour_length(tour)

# odwrócenie logiki -> dużo kilometrów = zły wynik, ale pierwszy na liście
def fitness_normalized (tour: list[int], tsp: TSPProblem, worst_score: int):
    return (worst_score - fitness(tour, tsp) + 1)

""" SELECTION """

def rank_select (population_in: list[list[int]], tsp: TSPProblem, rank_size: int
                 ) -> list[list[int]]:
    population = population_in[:]
    return (sorted(population, key=lambda tour: fitness(tour, tsp))
            [:rank_size])

def tournament_select (population: list[list[int]], tsp: TSPProblem,
                       tournament_size: int = 3) -> list[list[int]]:
    if not population:
        raise ValueError ("population is empty")
    if not (1 <= tournament_size <= len(population)):
        tournament_size = len(population)

    selected = []
    for _ in range(len(population)):
        contenders = random.sample(population, tournament_size)
        shortest_tour = min (contenders, key=lambda tour: fitness(tour, tsp))
        selected.append(shortest_tour)

    return selected

def roulette_select (population: list[list[int]], tsp: TSPProblem,
                     roulette_selection_size: int) -> list[list[int]]:
    if roulette_selection_size < 1 or roulette_selection_size > len(population):
        roulette_selection_size = len(population)

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
    child: [-1, -1, -1, -1, -1, -1, -1, -1, 8, 9]
    child: [-1, -1, -1, -1, -1, -1, -1, -1, 6, 4]
    Dziecko 1: [3, 7, 5, 1, 0, 2, 6, 4, 8, 9]
    Dziecko 2: [0, 1, 2, 3, 5, 7, 8, 9, 6, 4]
    """
    start = random.randint(0,parent_size-1)
    end = random.randint(start+1,parent_size)

    def make_a_child_ox(pA: list[int], pB:list[int]) -> list[int]:
        child = [-1] * parent_size
        child[start:end] = pA[start:end]
        taken = set(child[start:end])

        # modulo "przewija" indeksy od początku listy
        index = end % parent_size
        for i in range (parent_size):
            gene = pB[(end + i) % parent_size]
            if gene in taken:
                continue
            while child[index] != -1:
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
    child: [-1, -1, -1, -1, -1, -1, -1, -1, 8, 9]
    child: [-1, -1, -1, -1, -1, -1, -1, -1, 6, 4]
    Dziecko 1: [3, 7, 5, 1, 4, 0, 2, 6, 8, 9]
    Dziecko 2: [0, 1, 2, 3, 9, 5, 8, 7, 6, 4]
    """

    start = random.randint(0,parent_size-1)
    end = random.randint(start+1,parent_size)

    def lets_make_a_baby_pmx(pA: list[int], pB:list[int]) -> list[int]:
        child = [-1] * parent_size
        child[start:end] = pA[start:end]

        for i in range (start, end):
            gene_from_pB = pB[i]

            if gene_from_pB in child[start:end]:
                continue
            
            position = i
            while child[position] != -1:
                value_in_child = child[position]
                position = pB.index(value_in_child)

            child[position] = gene_from_pB

        for i in range (parent_size):
            if child[i] == -1:
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

        def find_neighbours (parent):
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

def swap_mutation (tour_in: list[int], mutation_probability: float = 0.1) -> list[int]:
    tour = tour_in[:]
    if random.random() < mutation_probability:
        pick1 = random.randint(0, len(tour)-1)
        pick2 = random.randint(0, len(tour)-1)
        tour [pick1], tour[pick2] = tour[pick2], tour[pick1]
    return tour

def inversion_mutation (tour_in: list[int], mutation_probability: float = 0.1) -> list[int]:
    tour = tour_in[:]
    if random.random() < mutation_probability:
        tour_size = len(tour)
        if tour_size < 2:
            return tour
        
        start = random.randint(0,tour_size-1)
        end = random.randint(start+1, tour_size)

        tour[start:end] = reversed(tour[start:end])

    return tour

def scramble_mutation (tour_in: list[int], mutation_probability: float = 0.1) -> list[int]:
    tour = tour_in[:]
    if random.random() < mutation_probability:
        tour_size = len(tour)
        if tour_size < 2:
            return tour
        
        start = random.randint(0,tour_size-1)
        end = random.randint(start+1, tour_size)

        segment = tour[start:end]
        random.shuffle(segment)
        tour[start:end] = segment

    return tour