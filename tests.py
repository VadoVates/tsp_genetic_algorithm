"""
Kompletny zestaw testów jednostkowych dla operators.py
Testy sprawdzają poprawność działania wszystkich funkcji oraz wykrywają edge case'y
"""

"""
test.py - Testy jednostkowe i integracyjne dla algorytmu genetycznego TSP
"""

import pytest
import random
from tsp_problem import TSPProblem
from operators import (
    initialize_population, fitness, rank_select, 
    tournament_select, roulette_select,
    order_crossover, partially_mapped_crossover, edge_recombination_crossover,
    swap_mutation, inversion_mutation, scramble_mutation
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def tsp():
    """Fixture: załaduj problem TSP"""
    return TSPProblem("data/att48.tsp")


@pytest.fixture
def cities(tsp):
    """Fixture: lista miast"""
    return list(tsp.coordinates.keys())


@pytest.fixture
def sample_population(cities):
    """Fixture: przykładowa populacja 10 tras"""
    random.seed(42)
    return initialize_population(cities, 10)


@pytest.fixture
def sample_tour(cities):
    """Fixture: pojedyncza trasa"""
    random.seed(42)
    tour = cities[:]
    random.shuffle(tour)
    return tour


# ============================================
# TESTY INICJALIZACJI
# ============================================

class TestInitialization:
    
    def test_initialize_population_size(self, cities):
        """Test: Populacja ma właściwy rozmiar"""
        population = initialize_population(cities, 20)
        assert len(population) == 20
    
    def test_initialize_population_is_permutation(self, cities):
        """Test: Każda trasa jest permutacją miast"""
        population = initialize_population(cities, 10)
        for tour in population:
            assert sorted(tour) == sorted(cities)
    
    def test_initialize_population_no_duplicates_in_tour(self, cities):
        """Test: Brak duplikatów w trasie"""
        population = initialize_population(cities, 10)
        for tour in population:
            assert len(tour) == len(set(tour))


# ============================================
# TESTY FITNESS
# ============================================

class TestFitness:
    
    def test_fitness_returns_number(self, sample_tour, tsp):
        """Test: Fitness zwraca liczbę"""
        result = fitness(sample_tour, tsp)
        assert isinstance(result, (int, float))
        assert result > 0
    
    def test_fitness_same_tour_same_result(self, sample_tour, tsp):
        """Test: Ten sam tour daje ten sam fitness"""
        result1 = fitness(sample_tour, tsp)
        result2 = fitness(sample_tour, tsp)
        assert result1 == result2
    
    def test_fitness_optimal_vs_random(self, tsp):
        """Test: Losowa trasa jest gorsza niż optymalna"""
        random_tour = list(tsp.coordinates.keys())
        random.shuffle(random_tour)
        
        optimal_tour = tsp.optimal_tour("data/att48.opt.tour")
        
        random_fitness = fitness(random_tour, tsp)
        optimal_fitness = fitness(optimal_tour, tsp)
        
        assert random_fitness > optimal_fitness  # Większa = gorsza dla TSP


# ============================================
# TESTY SELEKCJI
# ============================================

class TestSelection:
    
    def test_rank_select_returns_list(self, sample_population, tsp):
        """Test: rank_select zwraca listę"""
        result = rank_select(sample_population, tsp, 5)
        assert isinstance(result, list)
        assert len(result) == 5
    
    def test_rank_select_returns_best(self, sample_population, tsp):
        """Test: rank_select zwraca najlepszych"""
        result = rank_select(sample_population, tsp, 3)
        
        # Sprawdź czy są posortowani
        fitness_values = [fitness(tour, tsp) for tour in result]
        assert fitness_values == sorted(fitness_values)
    
    def test_tournament_select_returns_single_tour(self, sample_population, tsp):
        """Test: tournament_select zwraca pojedynczą trasę w liście"""
        result = tournament_select(sample_population, tsp, 3)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
    
    def test_tournament_select_is_from_population(self, sample_population, tsp):
        """Test: tournament_select zwraca trasę z populacji"""
        result = tournament_select(sample_population, tsp, 3)
        winner = result[0]
        
        # Zwycięzca musi być jednym z osobników z populacji
        assert sorted(winner) == sorted(sample_population[0])
    
    def test_roulette_select_returns_list(self, sample_population, tsp):
        """Test: roulette_select zwraca listę"""
        result = roulette_select(sample_population, tsp, 8)
        assert isinstance(result, list)
        assert len(result) == 8
    
    def test_roulette_select_valid_size(self, sample_population, tsp):
        """Test: roulette_select z nieprawidłowym rozmiarem rzuca wyjątek"""
        with pytest.raises(ValueError):
            roulette_select(sample_population, tsp, 0)
        
        with pytest.raises(ValueError):
            roulette_select(sample_population, tsp, 100)


# ============================================
# TESTY KRZYŻOWANIA
# ============================================

class TestCrossover:
    
    @pytest.mark.parametrize("crossover_func", [
        order_crossover,
        partially_mapped_crossover,
        edge_recombination_crossover
    ])
    def test_crossover_returns_two_children(self, sample_tour, cities, crossover_func):
        """Test: Crossover zwraca dwoje dzieci"""
        parent2 = cities[:]
        random.shuffle(parent2)
        
        child1, child2 = crossover_func(sample_tour, parent2)
        
        assert isinstance(child1, list)
        assert isinstance(child2, list)
    
    @pytest.mark.parametrize("crossover_func", [
        order_crossover,
        partially_mapped_crossover,
        edge_recombination_crossover
    ])
    def test_crossover_children_are_permutations(self, sample_tour, cities, crossover_func):
        """Test: Dzieci są poprawnymi permutacjami"""
        parent2 = cities[:]
        random.shuffle(parent2)
        
        child1, child2 = crossover_func(sample_tour, parent2)
        
        assert sorted(child1) == sorted(cities), f"Child1 nie jest permutacją: {child1}"
        assert sorted(child2) == sorted(cities), f"Child2 nie jest permutacją: {child2}"
    
    @pytest.mark.parametrize("crossover_func", [
        order_crossover,
        partially_mapped_crossover,
        edge_recombination_crossover
    ])
    def test_crossover_no_duplicates(self, sample_tour, cities, crossover_func):
        """Test: Brak duplikatów w dzieciach"""
        parent2 = cities[:]
        random.shuffle(parent2)
        
        child1, child2 = crossover_func(sample_tour, parent2)
        
        assert len(child1) == len(set(child1)), f"Child1 ma duplikaty"
        assert len(child2) == len(set(child2)), f"Child2 ma duplikaty"


# ============================================
# TESTY MUTACJI
# ============================================

class TestMutation:
    
    @pytest.mark.parametrize("mutation_func", [
        swap_mutation,
        inversion_mutation,
        scramble_mutation
    ])
    def test_mutation_probability_zero(self, sample_tour, mutation_func):
        """Test: Mutacja z prob=0 nie zmienia trasy"""
        original = sample_tour[:]
        mutated = mutation_func(sample_tour, 0.0)
        assert mutated == original
    
    @pytest.mark.parametrize("mutation_func", [
        swap_mutation,
        inversion_mutation,
        scramble_mutation
    ])
    def test_mutation_preserves_permutation(self, sample_tour, cities, mutation_func):
        """Test: Mutacja zachowuje permutację"""
        mutated = mutation_func(sample_tour, 1.0)
        assert sorted(mutated) == sorted(cities)
    
    @pytest.mark.parametrize("mutation_func", [
        swap_mutation,
        inversion_mutation,
        scramble_mutation
    ])
    def test_mutation_probability_one_changes_tour(self, sample_tour, mutation_func):
        """Test: Mutacja z prob=1.0 zmienia trasę (probabilistycznie)"""
        original = sample_tour[:]
        mutations_detected = 0
        
        # Wykonaj 10 prób
        for _ in range(10):
            mutated = mutation_func(original[:], 1.0)
            if mutated != original:
                mutations_detected += 1
        
        # Przynajmniej 8/10 powinno się zmienić
        assert mutations_detected >= 8, f"Tylko {mutations_detected}/10 mutacji wykryto"
    
    def test_mutation_probability_statistics(self, sample_tour):
        """Test: Prawdopodobieństwo mutacji działa statystycznie"""
        mutation_prob = 0.3
        mutations = 0
        trials = 1000
        
        for _ in range(trials):
            original = sample_tour[:]
            mutated = swap_mutation(original, mutation_prob)
            if mutated != original:
                mutations += 1
        
        observed_prob = mutations / trials
        # Sprawdź czy mieści się w 20% marginesu błędu
        assert abs(observed_prob - mutation_prob) < 0.05, \
            f"Oczekiwano ~{mutation_prob}, otrzymano {observed_prob}"


# ============================================
# TESTY INTEGRACYJNE - JEDNA GENERACJA
# ============================================

class TestOneGeneration:
    
    def test_full_generation_preserves_population_size(self, tsp, cities):
        """Test: Pełna generacja zachowuje rozmiar populacji"""
        population_size = 20
        elite_size = 2
        
        # Inicjalizacja
        population = initialize_population(cities, population_size)
        
        # Sortowanie
        population_sorted = rank_select(population, tsp, population_size)
        
        # Elityzm
        elites = population_sorted[:elite_size]
        remaining_count = population_size - elite_size
        
        # Selekcja
        parents = []
        while len(parents) < remaining_count:
            parents.extend(tournament_select(population_sorted, tsp, 3))
        
        # Krzyżowanie
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = order_crossover(
                parents[i % len(parents)], 
                parents[(i + 1) % len(parents)]
            )
            offspring.extend([child1, child2])
        
        # Mutacja
        mutated = [swap_mutation(tour, 0.1) for tour in offspring]
        
        # Nowa populacja
        new_population = elites + rank_select(mutated, tsp, remaining_count)
        
        assert len(new_population) == population_size
    
    def test_elitism_preserves_best(self, tsp, cities):
        """Test: Elityzm zachowuje najlepszą trasę"""
        population_size = 20
        elite_size = 1
        
        # Inicjalizacja
        population = initialize_population(cities, population_size)
        
        # Znajdź najlepszą trasę w starej populacji
        population_sorted = rank_select(population, tsp, population_size)
        best_old = population_sorted[0]
        best_old_fitness = fitness(best_old, tsp)
        
        # Pełna generacja
        elites = population_sorted[:elite_size]
        remaining_count = population_size - elite_size
        
        parents = []
        while len(parents) < remaining_count:
            parents.extend(tournament_select(population_sorted, tsp, 3))
        
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = order_crossover(
                parents[i % len(parents)], 
                parents[(i + 1) % len(parents)]
            )
            offspring.extend([child1, child2])
        
        mutated = [swap_mutation(tour, 0.1) for tour in offspring]
        new_population = elites + rank_select(mutated, tsp, remaining_count)
        
        # Znajdź najlepszą trasę w nowej populacji
        new_population_sorted = rank_select(new_population, tsp, population_size)
        best_new = new_population_sorted[0]
        best_new_fitness = fitness(best_new, tsp)
        
        # Najlepsza nowa trasa nie może być gorsza niż najlepsza stara
        assert best_new_fitness <= best_old_fitness


# ============================================
# TESTY WYDAJNOŚCIOWE (opcjonalne)
# ============================================

class TestPerformance:
    
    @pytest.mark.slow
    def test_100_generations_completes(self, tsp, cities):
        """Test: 100 generacji kończy się bez błędu"""
        population_size = 50
        elite_size = 5
        generations = 100
        
        population = initialize_population(cities, population_size)
        
        for gen in range(generations):
            population_sorted = rank_select(population, tsp, population_size)
            elites = population_sorted[:elite_size]
            remaining_count = population_size - elite_size
            
            parents = []
            while len(parents) < remaining_count:
                parents.extend(tournament_select(population_sorted, tsp, 3))
            
            offspring = []
            for i in range(0, len(parents), 2):
                child1, child2 = order_crossover(
                    parents[i % len(parents)], 
                    parents[(i + 1) % len(parents)]
                )
                offspring.extend([child1, child2])
            
            mutated = [swap_mutation(tour, 0.1) for tour in offspring]
            population = elites + rank_select(mutated, tsp, remaining_count)
            
            assert len(population) == population_size
        
        # Sprawdź czy znaleziono lepsze rozwiązanie niż losowe
        best = rank_select(population, tsp, 1)[0]
        best_fitness = fitness(best, tsp)
        
        random_tour = cities[:]
        random.shuffle(random_tour)
        random_fitness = fitness(random_tour, tsp)
        
        assert best_fitness < random_fitness


# ============================================
# TESTY DEBUGOWANIA - VERBOSE OUTPUT
# ============================================

@pytest.mark.verbose
def test_one_generation_verbose(tsp, cities, capsys):
    """Test z pełnym outputem - do debugowania"""
    population_size = 10
    elite_size = 2
    
    print("\n" + "="*80)
    print("VERBOSE TEST - JEDNA GENERACJA")
    print("="*80)
    
    # Inicjalizacja
    population = initialize_population(cities, population_size)
    print(f"\n1. Populacja: {len(population)} tras")
    
    # Sortowanie
    population_sorted = rank_select(population, tsp, population_size)
    fitness_values = [fitness(tour, tsp) for tour in population_sorted]
    print(f"2. Fitness: najlepszy={fitness_values[0]:.2f}, najgorszy={fitness_values[-1]:.2f}")
    
    # Elityzm
    elites = population_sorted[:elite_size]
    print(f"3. Elity: {len(elites)} tras")
    
    # Selekcja
    remaining_count = population_size - elite_size
    parents = []
    iterations = 0
    while len(parents) < remaining_count:
        parents.extend(tournament_select(population_sorted, tsp, 3))
        iterations += 1
    print(f"4. Selekcja: {len(parents)} rodziców w {iterations} iteracjach")
    
    # Krzyżowanie
    offspring = []
    for i in range(0, len(parents), 2):
        child1, child2 = order_crossover(
            parents[i % len(parents)], 
            parents[(i + 1) % len(parents)]
        )
        offspring.extend([child1, child2])
    print(f"5. Krzyżowanie: {len(offspring)} potomków")
    
    # Mutacja
    mutated = [swap_mutation(tour, 0.3) for tour in offspring]
    mutations = sum(1 for orig, mut in zip(offspring, mutated) if orig != mut)
    print(f"6. Mutacja: {mutations}/{len(mutated)} zmutowanych")
    
    # Nowa populacja
    new_population = elites + rank_select(mutated, tsp, remaining_count)
    print(f"7. Nowa populacja: {len(new_population)} tras")
    
    print("="*80 + "\n")
    
    assert len(new_population) == population_size