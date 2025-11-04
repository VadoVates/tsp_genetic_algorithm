"""
ROZWIĄZANIE PRZYKŁADOWEGO ZADANIA Z MODUŁU 2.
"""

import random

""" CODING / DECODING """

def pack (x1: int, x2: int) -> int:
    return (x1 << 5 | x2)

def unpack (chromosome: int) -> tuple[int, int]:
    x2 = chromosome & 0b11111
    x1 = (chromosome >> 5) & 0b11111
    return x1, x2

""" FITNESS / FUNKCJA CELU """

def f (chromosome: int) -> int:
    x1, x2 = unpack (chromosome)
    return x1 - x2

def g (chromosome: int) -> int:
    return f (chromosome) + 32

""" RANK SELECTION """
def rank_select (population: list[int], population_size: int) -> list[int]:
    return (sorted(population, key=g, reverse=True) [:population_size])

""" TOURNAMENT SELECTION """
def tournament_select (population: list[int], tournament_size: int = 3) -> int:
    if not population:
        raise ValueError ("Population is empty")
    if not (1 <= tournament_size <= len(population)):
        raise ValueError("Invalid tournament size")

    contenders = random.sample(population, tournament_size)
    winner = max (contenders, key=g)
    return winner

""" ROULETTE SELECTION """

def roulette_select (population: list[int]) -> list[int]:
    fitness_list = [g(chromosome) for chromosome in population]
    total = sum (fitness_list)

    if total == 0: # jeżeli wszystkie dają różnicę zero, wtedy wybierz losowo
        return random.choices (population, k = len(population))

    fitness_accumulated = []
    s = 0
    for fit in fitness_list:
        s+=fit
        fitness_accumulated.append(s)

    selected = []
    for _ in range (len(population)):
        r = random.randint (1, total)
        for i, accumulation in enumerate (fitness_accumulated):
            if r <= accumulation:
                selected.append(population[i])
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

def one_point_crossover (population: list[int], cross_propability: float = 0.5
                         ) -> list[int]:
    sh = population[:] # klonowanie
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

def mutate (chromosome: int, mutation_propability: float = 0.02) -> int:
    c = chromosome
    for i in range(10):
        if random.random() < mutation_propability:
            # podmiana 1 bitu przez XOR
            c = c ^ (1 << i)

    return c

def mutate_population (population: list[int], mutation_propability: float = 0.02
                       ) -> list[int]:
    out = []
    for chromosome in population:
        out.append(mutate (chromosome, mutation_propability))
    return out

""" DESCRIBE VALUES """

def print_binary (value: int) -> str:
    return format(value & 0x3FF, '010b')

def describe_population (population: list[int]) -> str:
    lines = []
    s = 0
    for i, chromosome in enumerate(population, 1):
        x1, x2 = unpack(chromosome)
        fc, gc = f(chromosome), g(chromosome)
        s += gc
        lines.append(
            f"{i:>2}. bits={print_binary(chromosome)} (x1,x2)=({x1:>2},{x2:>2})  f={fc:>3}  g={gc:>3}")
    lines.append(f"sum g = {s}")
    return "\n".join(lines)

""" MAIN ALGORITHM """

def genetic_algorithm (
        population: list[int],
        generations: int = 100,
        cross_propability: float = 0.5,
        mutation_propability: float = 0.02
    ):
    best_overall = max (population, key=g)
    
    for gen in range(generations):
        # 1. Reprodukcja ruletką:
        selected = roulette_select (population)
        # 2. Krzyżowanie:
        crossed = one_point_crossover (selected, cross_propability = cross_propability)
        # 3. Mutacja:
        mutated = mutate_population (crossed, mutation_propability=mutation_propability)
        # 4. Rank crossover:
        population = rank_select(mutated, len(population))

    best = max (population, key=g)
    x1, x2 = unpack (best)

    print (f"Najlepszy (x1, x2): ({x1}, {x2}). f = {f(best)}, g = {g(best)})")

if __name__ == "__main__":
    population = [
        0b1000110000,
        0b1010110101,
        0b0010100101,
        0b0000100001,
        0b0001100011
    ]
    genetic_algorithm(population, generations = 100, cross_propability = 0.75, mutation_propability = 0.02)