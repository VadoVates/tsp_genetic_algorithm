import random, bisect

""" INICJALIZACJA POPULACJI """

def create_random_permutation (cities: list[int]) -> list[int]:
    tour = cities[:] #kopia
    random.shuffle(tour)
    return tour

def initialize_population (cities: list[int], population_size: int) -> list[list[int]]:
    return [create_random_permutation(cities) for _ in range (population_size)]

""" SELECTION """

def is_data_ok (population_in: list[list[int]], parameter: int, num_select: int):
    if not population_in:
        raise ValueError("Population must be initialized")
    if parameter < 1 or parameter > len(population_in):
        raise ValueError("Parameter out of range")
    if num_select < 1:
        raise ValueError("Number of selection out of range")

def rank_select (population_in: list[list[int]], fitness_cache: dict[int, int], rank_size: int,
                 num_select: int = 1) -> list[list[int]]:
    is_data_ok (population_in, rank_size, num_select)

    # 1) top rank_size (ranking by fitness)
    ranked = sorted(population_in, key=lambda tour: fitness_cache[id(tour)])
    pool = ranked[:rank_size]  # najlepszy na index 0

    # wagi po randze: najlepszy ma największą wagę = rank_size, najgorszy = 1
    cum = []
    acc = 0
    for w in range(rank_size, 0, -1):
        acc += w
        cum.append(acc)
    total = cum[-1]

    selected = []
    for _ in range(num_select):
        r = random.randint(1, total)
        i = bisect.bisect_left(cum, r)
        selected.append(pool[i])
    return selected

def tournament_select (population_in: list[list[int]], fitness_cache: dict[int, int],
                       tournament_size: int = 3, num_select: int = 1) -> list[list[int]]:
    is_data_ok (population_in, tournament_size, num_select)

    winners: list[list[int]] = []
    for _ in range (num_select):
        contenders = random.sample(population_in, tournament_size)
        winner = min (contenders, key=lambda tour: fitness_cache[id(tour)])
        winners.append(winner)
    return winners

def roulette_select (population_in: list[list[int]], fitness_cache: dict[int, int],
                     roulette_selection_size: int, num_select: int = 1) -> list[list[int]]:
    is_data_ok(population_in, roulette_selection_size, num_select)

    selected: list[list[int]] = []

    for _ in range (num_select):
        contenders = random.sample(population_in, roulette_selection_size)

        scores = [fitness_cache[id(tour)] for tour in contenders]
        worst = max(scores)

        weights = [(worst - score) + 1 for score in scores]

        total = 0
        accumulated: list[int] = []
        for weight in weights:
            total += weight
            accumulated.append(total)

        r = random.randint(1, total)
        i = bisect.bisect_left(accumulated, r)
        selected.append(contenders[i])

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

    def make_a_child_ox(par_a: list[int], par_b:list[int]) -> list[int]:
        child = [-1] * parent_size
        child[start:end] = par_a[start:end]
        taken = set(child[start:end])

        # modulo "przewija" indeksy od początku listy
        index = end % parent_size
        for i in range (parent_size):
            gene = par_b[(end + i) % parent_size]
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

    def lets_make_a_baby_pmx(par_a: list[int], par_b:list[int]) -> list[int]:
        child = [-1] * parent_size
        child[start:end] = par_a[start:end]

        for i in range (start, end):
            gene_from_par_b = par_b[i]

            if gene_from_par_b in child[start:end]:
                continue

            position = i
            while child[position] != -1:
                value_in_child = child[position]
                position = par_b.index(value_in_child)

            child[position] = gene_from_par_b

        for i in range (parent_size):
            if child[i] == -1:
                child[i] = par_b[i]

        return child

    child1 = lets_make_a_baby_pmx(parent1, parent2)
    child2 = lets_make_a_baby_pmx(parent2, parent1)
    return child1, child2

def edge_recombination_crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    parent_size = len(parent1)
    assert len(parent2) == parent_size and len(set(parent1)) == parent_size and len(set(parent2)) == parent_size

    def build_edge_table(par_a: list[int], par_b: list[int]) -> dict[int, set[int]]:
        # utwórz puste zbiory z kluczem każdego miasta
        edge_table = {city: set() for city in par_a}

        def find_neighbours (parent):
            for i in range(parent_size):
                current = parent[i]
                prev_city = parent[(i-1) % parent_size]
                next_city = parent[(i+1) % parent_size]
                edge_table[current].add(prev_city)
                edge_table[current].add(next_city)
        
        find_neighbours(par_a)
        find_neighbours(par_b)

        return edge_table
    
    def give_me_a_child_erx(par_a: list[int], par_b: list[int]) -> list[int]:
        edge_table = build_edge_table (par_a, par_b)
        child: list[int] = []

        # bierzemy randomowe miasto z pierwszego ojca/matuli
        current = random.choice(par_a)
        while len(child) < parent_size:
            child.append(current)

            # wyciepujemy `current` z wszystkich list sąsiedzkich
            for city in edge_table:
                edge_table[city].discard(current)

            if not edge_table[current]:
                remaining = [c for c in par_a if c not in child]
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