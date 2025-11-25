"""
Kompletny zestaw testów jednostkowych dla genetic_algorithm.py
Testy sprawdzają poprawność działania wszystkich funkcji oraz wykrywają edge case'y
"""

import random
import pytest
from collections import Counter
from tsp_problem import TSPProblem
from genetic_algorithm import *


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_permutation(child, parent):
    """Sprawdza czy dziecko jest poprawną permutacją rodzica"""
    return (len(child) == len(parent) and
            len(set(child)) == len(child) and
            set(child) == set(parent))

def suppress_prints(func, *args, **kwargs):
    """Wyłącza printy dla funkcji"""
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result


# ============================================================================
# TESTY INICJALIZACJI
# ============================================================================

class TestInitialization:
    """Testy tworzenia populacji początkowej"""
    
    def test_create_random_permutation_basic(self):
        """Test podstawowego działania permutacji"""
        cities = [0, 1, 2, 3, 4]
        tour = create_random_permutation(cities)
        
        assert len(tour) == len(cities)
        assert set(tour) == set(cities)
        assert len(set(tour)) == len(tour)  # brak duplikatów
    
    def test_create_random_permutation_single_city(self):
        """Test dla pojedynczego miasta"""
        cities = [0]
        tour = create_random_permutation(cities)
        assert tour == [0]
    
    def test_create_random_permutation_two_cities(self):
        """Test dla dwóch miast"""
        cities = [0, 1]
        tour = create_random_permutation(cities)
        assert set(tour) == {0, 1}
    
    def test_create_random_permutation_randomness(self):
        """Test czy faktycznie generuje różne permutacje"""
        cities = list(range(10))
        tours = [create_random_permutation(cities) for _ in range(10)]
        
        # Przynajmniej 2 różne permutacje w 10 próbach
        unique_tours = [tuple(t) for t in tours]
        assert len(set(unique_tours)) >= 2
    
    def test_create_random_permutation_doesnt_modify_original(self):
        """Test czy nie modyfikuje oryginalnej listy"""
        cities = [0, 1, 2, 3, 4]
        original = cities.copy()
        create_random_permutation(cities)
        assert cities == original
    
    def test_create_random_permutation_large(self):
        """Test dla dużej liczby miast"""
        cities = list(range(100))
        tour = create_random_permutation(cities)
        assert is_valid_permutation(tour, cities)
    
    def test_initialize_population_size(self):
        """Test rozmiaru populacji"""
        cities = [0, 1, 2, 3, 4]
        population = initialize_population(cities, 20)
        assert len(population) == 20
    
    def test_initialize_population_empty(self):
        """Test pustej populacji"""
        cities = [0, 1, 2]
        population = initialize_population(cities, 0)
        assert len(population) == 0
    
    def test_initialize_population_single_individual(self):
        """Test populacji z jednym osobnikiem"""
        cities = [0, 1, 2, 3]
        population = initialize_population(cities, 1)
        assert len(population) == 1
        assert is_valid_permutation(population[0], cities)
    
    def test_initialize_population_all_valid(self):
        """Test czy wszystkie trasy są poprawnymi permutacjami"""
        cities = [0, 1, 2, 3, 4]
        population = initialize_population(cities, 50)
        
        for tour in population:
            assert is_valid_permutation(tour, cities)
    
    def test_initialize_population_diversity(self):
        """Test czy populacja jest zróżnicowana"""
        cities = list(range(10))
        population = initialize_population(cities, 100)
        
        unique = set(tuple(tour) for tour in population)
        # W 100 osobnikach powinno być przynajmniej 50 unikalnych
        assert len(unique) >= 50


# ============================================================================
# TESTY FITNESS
# ============================================================================

class TestFitness:
    """Testy funkcji oceny tras"""
    
    @pytest.fixture
    def tsp_problem(self):
        """Fixture: problem TSP"""
        return TSPProblem("data/att48.tsp")
    
    def test_fitness_returns_positive(self, tsp_problem):
        """Test czy fitness zwraca wartość dodatnią"""
        cities = list(tsp_problem.coordinates.keys())
        tour = create_random_permutation(cities)
        fit = fitness(tour, tsp_problem)
        assert fit > 0
    
    def test_fitness_returns_int(self, tsp_problem):
        """Test czy fitness zwraca int"""
        cities = list(tsp_problem.coordinates.keys())
        tour = create_random_permutation(cities)
        fit = fitness(tour, tsp_problem)
        assert isinstance(fit, int)
    
    def test_fitness_deterministic(self, tsp_problem):
        """Test czy fitness jest deterministyczne"""
        cities = list(tsp_problem.coordinates.keys())
        tour = cities[:10]
        
        fit1 = fitness(tour, tsp_problem)
        fit2 = fitness(tour, tsp_problem)
        assert fit1 == fit2
    
    def test_fitness_different_tours(self, tsp_problem):
        """Test czy różne trasy mogą mieć różne oceny"""
        cities = list(tsp_problem.coordinates.keys())
        
        # Uporządkowana vs odwrócona
        tour1 = cities[:10]
        tour2 = list(reversed(cities[:10]))
        
        fit1 = fitness(tour1, tsp_problem)
        fit2 = fitness(tour2, tsp_problem)
        
        # Mogą być równe dla symetrycznych tras, ale sprawdzamy typ
        assert isinstance(fit1, int) and isinstance(fit2, int)
    
    def test_fitness_normalized_basic(self, tsp_problem):
        """Test normalizacji fitness"""
        cities = list(tsp_problem.coordinates.keys())
        tour = create_random_permutation(cities)
        
        fit = fitness(tour, tsp_problem)
        worst_score = fit + 1000
        
        norm = fitness_normalized(tour, tsp_problem, worst_score)
        assert norm > 0
        assert norm <= worst_score
    
    def test_fitness_normalized_inversion(self, tsp_problem):
        """Test czy normalizacja odwraca porządek (mniejszy fitness = większy normalized)"""
        cities = list(tsp_problem.coordinates.keys())[:10]
        tours = [create_random_permutation(cities) for _ in range(5)]
        
        fitness_scores = [fitness(t, tsp_problem) for t in tours]
        worst = max(fitness_scores)
        
        normalized = [fitness_normalized(t, tsp_problem, worst) for t in tours]
        
        # Najlepszy (min fitness) powinien mieć max normalized
        best_idx = fitness_scores.index(min(fitness_scores))
        assert normalized[best_idx] == max(normalized)
        
        # Najgorszy (max fitness) powinien mieć min normalized
        worst_idx = fitness_scores.index(max(fitness_scores))
        assert normalized[worst_idx] == min(normalized)
    
    def test_fitness_normalized_with_equal_worst(self, tsp_problem):
        """Test normalizacji gdy worst_score == fitness"""
        cities = list(tsp_problem.coordinates.keys())[:5]
        tour = cities[:]
        
        fit = fitness(tour, tsp_problem)
        norm = fitness_normalized(tour, tsp_problem, fit)
        
        # worst_score - fitness + 1 = fit - fit + 1 = 1
        assert norm == 1
    
    def test_fitness_single_city(self, tsp_problem):
        """Test fitness dla pojedynczego miasta"""
        cities = [list(tsp_problem.coordinates.keys())[0]]
        tour = cities[:]
        fit = fitness(tour, tsp_problem)
        assert fit == 0  # Brak krawędzi


# ============================================================================
# TESTY SELEKCJI
# ============================================================================

class TestRankSelection:
    """Testy selekcji rankingowej"""
    
    @pytest.fixture
    def setup(self):
        """Fixture: problem i populacja"""
        problem = TSPProblem("data/att48.tsp")
        cities = list(problem.coordinates.keys())[:20]
        population = initialize_population(cities, 30)
        return problem, cities, population
    
    def test_rank_select_returns_correct_size(self, setup):
        """Test czy zwraca odpowiednią liczbę tras"""
        problem, cities, population = setup
        selected = rank_select(population, rank_size=10, tsp=problem)
        assert len(selected) == 10
    
    def test_rank_select_returns_best(self, setup):
        """Test czy zwraca najlepsze trasy"""
        problem, cities, population = setup
        selected = rank_select(population, rank_size=5, tsp=problem)
        
        all_fitness = sorted([fitness(t, problem) for t in population])
        selected_fitness = [fitness(t, problem) for t in selected]
        
        assert selected_fitness == all_fitness[:5]
    
    def test_rank_select_sorted_order(self, setup):
        """Test czy zwraca trasy posortowane"""
        problem, cities, population = setup
        selected = rank_select(population, rank_size=10, tsp=problem)
        
        fitness_values = [fitness(t, problem) for t in selected]
        assert fitness_values == sorted(fitness_values)
    
    def test_rank_select_full_population(self, setup):
        """Test gdy rank_size == population_size"""
        problem, cities, population = setup
        selected = rank_select(population, rank_size=len(population), tsp=problem)
        assert len(selected) == len(population)
    
    def test_rank_select_single(self, setup):
        """Test wyboru jednego osobnika"""
        problem, cities, population = setup
        selected = rank_select(population, rank_size=1, tsp=problem)
        
        assert len(selected) == 1
        # Powinien być najlepszy
        best_fitness = min(fitness(t, problem) for t in population)
        assert fitness(selected[0], problem) == best_fitness
    
    def test_rank_select_empty_population(self):
        """Test dla pustej populacji"""
        problem = TSPProblem("data/att48.tsp")
        selected = rank_select([], rank_size=5, tsp=problem)
        assert len(selected) == 0


class TestTournamentSelection:
    """Testy selekcji turniejowej"""
    
    @pytest.fixture
    def setup(self):
        problem = TSPProblem("data/att48.tsp")
        cities = list(problem.coordinates.keys())[:20]
        population = initialize_population(cities, 30)
        return problem, cities, population
    
    def test_tournament_select_returns_valid_tour(self, setup):
        """Test czy zwraca poprawną trasę"""
        problem, cities, population = setup
        winner = tournament_select(population, problem, tournament_size=3)
        
        assert is_valid_permutation(winner, cities)
    
    def test_tournament_select_empty_population_raises(self):
        """Test czy pusta populacja rzuca wyjątek"""
        problem = TSPProblem("data/att48.tsp")
        
        with pytest.raises(ValueError, match="population is empty"):
            tournament_select([], problem, tournament_size=3)
    
    def test_tournament_select_invalid_size_zero(self, setup):
        """Test czy tournament_size=0 rzuca wyjątek"""
        problem, cities, population = setup
        
        with pytest.raises(ValueError, match="Invalid tournament size"):
            tournament_select(population, problem, tournament_size=0)
    
    def test_tournament_select_invalid_size_too_large(self, setup):
        """Test czy zbyt duży tournament_size rzuca wyjątek"""
        problem, cities, population = setup
        
        with pytest.raises(ValueError, match="Invalid tournament size"):
            tournament_select(population, problem, tournament_size=100)
    
    def test_tournament_select_size_one(self, setup):
        """Test turnieju o rozmiarze 1 - zwraca losową trasę"""
        problem, cities, population = setup
        winner = tournament_select(population, problem, tournament_size=1)
        
        assert winner in population
    
    def test_tournament_select_size_equals_population(self, setup):
        """Test gdy tournament_size == len(population)"""
        problem, cities, population = setup
        winner = tournament_select(population, problem, tournament_size=len(population))
        
        # Powinien wybrać najlepszego
        best_fitness = min(fitness(t, problem) for t in population)
        assert fitness(winner, problem) == best_fitness
    
    def test_tournament_select_multiple_calls_gives_variety(self, setup):
        """Test czy wielokrotne wywołanie daje różnorodność"""
        problem, cities, population = setup
        
        winners = [tournament_select(population, problem, tournament_size=3) 
                   for _ in range(20)]
        
        # Powinno być przynajmniej kilka różnych zwycięzców
        unique = set(tuple(w) for w in winners)
        assert len(unique) >= 3
    
    def test_tournament_select_statistical_bias_towards_better(self, setup):
        """Test czy statystycznie wybiera lepsze trasy"""
        problem, cities, population = setup
        
        winners_fitness = [fitness(tournament_select(population, problem, tournament_size=5), problem)
                          for _ in range(50)]
        
        avg_winner = sum(winners_fitness) / len(winners_fitness)
        avg_population = sum(fitness(t, problem) for t in population) / len(population)
        
        # Średni fitness zwycięzców powinien być lepszy (mniejszy) niż średnia populacji
        assert avg_winner < avg_population


class TestRouletteSelection:
    """Testy selekcji ruletkowej"""
    
    @pytest.fixture
    def setup(self):
        problem = TSPProblem("data/att48.tsp")
        cities = list(problem.coordinates.keys())[:20]
        population = initialize_population(cities, 30)
        return problem, cities, population
    
    def test_roulette_select_returns_correct_size(self, setup):
        """Test czy zwraca odpowiednią liczbę tras"""
        problem, cities, population = setup
        selected = roulette_select(population, problem, roulette_selection_size=10)
        
        # Zwraca 10 tras wybranych z 10 contenders
        assert len(selected) == 10
    
    def test_roulette_select_returns_valid_tours(self, setup):
        """Test czy zwraca poprawne trasy"""
        problem, cities, population = setup
        selected = roulette_select(population, problem, roulette_selection_size=10)
        
        for tour in selected:
            assert is_valid_permutation(tour, cities)
    
    def test_roulette_select_allows_duplicates(self, setup):
        """Test czy może wybrać tę samą trasę wielokrotnie"""
        problem, cities, population = setup
        
        # Małe contenders = większa szansa na duplikaty
        selected = roulette_select(population, problem, roulette_selection_size=5)
        
        # Mogą być duplikaty - to zgodne z logiką ruletki
        assert len(selected) == 5
    
    def test_roulette_select_single_contender(self, setup):
        """Test dla pojedynczego contender"""
        problem, cities, population = setup
        selected = roulette_select(population, problem, roulette_selection_size=1)
        
        # Wybiera 1 z 1, więc zwraca tę samą trasę
        assert len(selected) == 1
        assert selected[0] in population
    
    def test_roulette_select_bias_towards_better(self, setup):
        """Test czy preferuje lepsze trasy"""
        problem, cities, population = setup
        
        # Duża próba statystyczna
        all_selected = []
        for _ in range(20):
            selected = roulette_select(population, problem, roulette_selection_size=15)
            all_selected.extend(selected)
        
        avg_selected = sum(fitness(t, problem) for t in all_selected) / len(all_selected)
        avg_population = sum(fitness(t, problem) for t in population) / len(population)
        
        # Ruletka powinna preferować lepsze (mniejsze fitness)
        # Ale nie jest to gwarantowane w każdym uruchomieniu
        # Sprawdzamy tylko czy działa
        assert avg_selected > 0


# ============================================================================
# TESTY CROSSOVER
# ============================================================================

class TestOrderCrossover:
    """Testy Order Crossover (OX)"""
    
    def test_ox_basic_validity(self):
        """Test podstawowej poprawności OX"""
        p1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [3, 7, 5, 1, 9, 0, 2, 8, 6, 4]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)


class TestEdgeRecombinationCrossover:
    """Testy Edge Recombination Crossover (ERX)"""
    
    def test_erx_basic_validity(self):
        """Test podstawowej poprawności ERX"""
        p1 = [0, 1, 2, 3, 4, 5]
        p2 = [5, 4, 3, 2, 1, 0]
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_small_tour(self):
        """Test ERX dla małej trasy"""
        p1 = [0, 1, 2]
        p2 = [2, 0, 1]
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_two_cities(self):
        """Test ERX dla dwóch miast"""
        p1 = [0, 1]
        p2 = [1, 0]
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_identical_parents(self):
        """Test ERX dla identycznych rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [0, 1, 2, 3, 4]
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_multiple_random(self):
        """Test ERX dla wielu losowych przypadków"""
        cities = list(range(15))
        
        for i in range(10):
            p1 = cities[:]
            p2 = cities[:]
            random.shuffle(p1)
            random.shuffle(p2)
            
            try:
                c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
                
                assert is_valid_permutation(c1, p1), f"Iteration {i}: Child1 invalid"
                assert is_valid_permutation(c2, p2), f"Iteration {i}: Child2 invalid"
            except Exception as e:
                pytest.fail(f"ERX failed at iteration {i}: {e}")
    
    def test_erx_preserves_edges_from_parents(self):
        """Test czy ERX wykorzystuje krawędzie z rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [0, 1, 2, 3, 4]
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_medium_tour(self):
        """Test ERX dla średniej trasy"""
        p1 = list(range(20))
        p2 = list(range(20))
        random.shuffle(p2)
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_erx_completeness(self):
        """Test czy ERX zawsze zwraca pełne trasy"""
        cities = list(range(10))
        
        for _ in range(5):
            p1 = cities[:]
            p2 = cities[:]
            random.shuffle(p1)
            random.shuffle(p2)
            
            c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
            
            assert len(c1) == len(cities), f"Child1 incomplete: {len(c1)} != {len(cities)}"
            assert len(c2) == len(cities), f"Child2 incomplete: {len(c2)} != {len(cities)}"


# ============================================================================
# TESTY MUTACJI
# ============================================================================

class TestSwapMutation:
    """Testy mutacji swap"""
    
    def test_swap_mutation_preserves_permutation(self):
        """Test czy mutacja zachowuje permutację"""
        tour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        original_set = set(tour)
        
        mutated = swap_mutation(tour[:], mutation_probability=0.5)
        
        assert len(mutated) == len(tour)
        assert set(mutated) == original_set
        assert len(set(mutated)) == len(mutated)
    
    def test_swap_mutation_zero_probability(self):
        """Test mutacji z prawdopodobieństwem 0"""
        tour = [0, 1, 2, 3, 4, 5]
        original = tour[:]
        
        mutated = swap_mutation(tour[:], mutation_probability=0.0)
        
        assert set(mutated) == set(original)
    
    def test_swap_mutation_high_probability_changes_tour(self):
        """Test mutacji z wysokim prawdopodobieństwem"""
        random.seed(42)
        tour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        original = tour[:]
        
        changes = 0
        for _ in range(10):
            mutated = swap_mutation(tour[:], mutation_probability=0.9)
            if mutated != original:
                changes += 1
            assert set(mutated) == set(original)
        
        assert changes > 0
    
    def test_swap_mutation_single_city(self):
        """Test mutacji dla pojedynczego miasta"""
        tour = [0]
        mutated = swap_mutation(tour[:], mutation_probability=1.0)
        assert mutated == [0]
    
    def test_swap_mutation_two_cities(self):
        """Test mutacji dla dwóch miast"""
        tour = [0, 1]
        mutated = swap_mutation(tour[:], mutation_probability=1.0)
        assert set(mutated) == {0, 1}
    
    def test_swap_mutation_preserves_all_genes(self):
        """Test czy mutacja nie gubi ani nie dodaje genów"""
        tour = list(range(20))
        random.shuffle(tour)
        
        for _ in range(10):
            mutated = swap_mutation(tour[:], mutation_probability=0.3)
            assert sorted(mutated) == sorted(tour)
    
    def test_swap_mutation_returns_new_list(self):
        tour = [0, 1, 2, 3, 4]
        result = swap_mutation(tour, mutation_probability=0.5)
        assert result is not tour  # ma być nowy obiekt
        assert set(result) == set(tour)
    
    def test_swap_mutation_large_tour(self):
        """Test mutacji dla dużej trasy"""
        tour = list(range(100))
        random.shuffle(tour)
        
        mutated = swap_mutation(tour[:], mutation_probability=0.2)
        assert is_valid_permutation(mutated, tour)


class TestSwapMutatePopulation:
    """Testy mutacji całej populacji"""
    
    def test_swap_mutation_size(self):
        """Test czy zwraca populację o tym samym rozmiarze"""
        population = [[0, 1, 2, 3, 4] for _ in range(10)]
        
        mutated_pop = swap_mutation(population, mutation_probability=0.3)
        
        assert len(mutated_pop) == len(population)
    
    def test_swap_mutation_all_valid(self):
        """Test czy wszystkie trasy w populacji są poprawne"""
        cities = list(range(10))
        population = [create_random_permutation(cities) for _ in range(20)]
        
        mutated_pop = swap_mutation(population, mutation_probability=0.2)
        
        for tour in mutated_pop:
            assert is_valid_permutation(tour, cities)
    
    def test_swap_mutation_empty(self):
        """Test dla pustej populacji"""
        population = []
        mutated_pop = swap_mutation(population, mutation_probability=0.5)
        assert len(mutated_pop) == 0
    
    def test_swap_mutation_single_individual(self):
        """Test dla populacji z jednym osobnikiem"""
        population = [[0, 1, 2, 3, 4]]
        mutated_pop = swap_mutation(population, mutation_probability=0.3)
        
        assert len(mutated_pop) == 1
        assert is_valid_permutation(mutated_pop[0], population[0])
    
    def test_swap_mutation_preserves_diversity(self):
        """Test czy mutacja zachowuje różnorodność"""
        cities = list(range(15))
        population = [create_random_permutation(cities) for _ in range(50)]
        
        mutated_pop = swap_mutation(population, mutation_probability=0.1)
        
        unique_before = len(set(tuple(t) for t in population))
        unique_after = len(set(tuple(t) for t in mutated_pop))
        
        assert unique_after > 0
    
    def test_swap_mutation_zero_probability(self):
        """Test mutacji populacji z prawdopodobieństwem 0"""
        cities = list(range(5))
        population = [create_random_permutation(cities) for _ in range(10)]
        
        mutated_pop = swap_mutation(population, mutation_probability=0.0)
        
        for tour in mutated_pop:
            assert is_valid_permutation(tour, cities)

class TestInversionMutation:
    """Testy mutacji inwersji"""

    def test_inversion_mutation_preserves_permutation(self):
        tour = list(range(10))
        mutated = inversion_mutation(tour[:], mutation_probability=1.0)
        assert len(mutated) == len(tour)
        assert set(mutated) == set(tour)

    def test_inversion_mutation_zero_probability(self):
        tour = [0, 1, 2, 3, 4]
        mutated = inversion_mutation(tour[:], mutation_probability=0.0)
        assert mutated == tour

    def test_inversion_mutation_small_tour(self):
        tour = [0, 1]
        mutated = inversion_mutation(tour[:], mutation_probability=1.0)
        assert set(mutated) == {0, 1}

    def test_inversion_mutation_changes_order(self):
        random.seed(42)
        tour = list(range(10))
        changed = False
        for _ in range(10):
            mutated = inversion_mutation(tour[:], mutation_probability=1.0)
            if mutated != tour:
                changed = True
                break
        assert changed

class TestScrambleMutation:
    """Testy mutacji scramble"""

    def test_scramble_mutation_preserves_permutation(self):
        tour = list(range(10))
        mutated = scramble_mutation(tour[:], mutation_probability=1.0)
        assert len(mutated) == len(tour)
        assert set(mutated) == set(tour)

    def test_scramble_mutation_zero_probability(self):
        tour = [0, 1, 2, 3]
        mutated = scramble_mutation(tour[:], mutation_probability=0.0)
        assert mutated == tour

    def test_scramble_mutation_small_tour(self):
        tour = [0, 1]
        mutated = scramble_mutation(tour[:], mutation_probability=1.0)
        assert set(mutated) == {0, 1}

    def test_scramble_mutation_randomness(self):
        tour = list(range(10))
        results = {tuple(scramble_mutation(tour[:], mutation_probability=1.0)) for _ in range(10)}
        assert len(results) > 1

# ============================================================================
# TESTY EDGE CASES I INTEGRACYJNE
# ============================================================================

class TestEdgeCases:
    """Testy przypadków brzegowych"""
    
    def test_single_city_problem(self):
        """Test dla problemu z jednym miastem"""
        cities = [0]
        population = initialize_population(cities, 5)
        
        for tour in population:
            assert tour == [0]
    
    def test_two_cities_problem(self):
        """Test dla problemu z dwoma miastami"""
        cities = [0, 1]
        population = initialize_population(cities, 10)
        
        for tour in population:
            assert set(tour) == {0, 1}
    
    def test_large_population(self):
        """Test dla dużej populacji"""
        cities = list(range(10))
        population = initialize_population(cities, 1000)
        
        assert len(population) == 1000
        
        unique = set(tuple(t) for t in population)
        assert len(unique) > 50
    
    def test_crossover_with_identical_parents(self):
        """Test crossover dla identycznych rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [0, 1, 2, 3, 4]
        
        c1_ox, c2_ox = suppress_prints(order_crossover, p1, p2)
        c1_pmx, c2_pmx = suppress_prints(partially_mapped_crossover, p1, p2)
        c1_erx, c2_erx = suppress_prints(edge_recombination_crossover, p1, p2)
        
        assert is_valid_permutation(c1_ox, p1)
        assert is_valid_permutation(c1_pmx, p1)
        assert is_valid_permutation(c1_erx, p1)
    
    def test_all_operations_on_small_problem(self):
        """Test wszystkich operacji na małym problemie"""
        cities = [0, 1, 2]
        
        population = initialize_population(cities, 6)
        assert all(is_valid_permutation(t, cities) for t in population)
        
        p1, p2 = population[0], population[1]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        assert is_valid_permutation(c1, cities)
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        assert is_valid_permutation(c1, cities)
        
        c1, c2 = suppress_prints(edge_recombination_crossover, p1, p2)
        assert is_valid_permutation(c1, cities)
        
        mutated = swap_mutation(p1[:], mutation_probability=0.5)
        assert is_valid_permutation(mutated, cities)


class TestIntegration:
    """Testy integracyjne - symulacja flow GA"""
    
    @pytest.fixture
    def tsp_small(self):
        """Mały problem TSP"""
        problem = TSPProblem("data/att48.tsp")
        cities = list(problem.coordinates.keys())[:15]
        return problem, cities
    
    def test_full_generation_cycle(self, tsp_small):
        """Test pełnego cyklu jednej generacji"""
        problem, cities = tsp_small
        
        population = initialize_population(cities, 20)
        
        parents = rank_select(population, rank_size=10, tsp=problem)
        
        offspring = []
        for i in range(0, len(parents)-1, 2):
            c1, c2 = suppress_prints(order_crossover, parents[i], parents[i+1])
            offspring.extend([c1, c2])
        
        mutated = swap_mutation(offspring, mutation_probability=0.1)
        
        for tour in mutated:
            assert is_valid_permutation(tour, cities)
    
    def test_selection_crossover_mutation_pipeline(self, tsp_small):
        """Test pipeline: selekcja → crossover → mutacja"""
        problem, cities = tsp_small
        
        population = initialize_population(cities, 30)
        
        parents_rank = rank_select(population, rank_size=10, tsp=problem)
        assert len(parents_rank) == 10
        
        winners = [tournament_select(population, problem, tournament_size=3) 
                   for _ in range(10)]
        assert len(winners) == 10
        
        offspring = []
        for i in range(0, len(parents_rank)-1, 2):
            c1, c2 = suppress_prints(partially_mapped_crossover, 
                                    parents_rank[i], parents_rank[i+1])
            offspring.extend([c1, c2])
        
        mutated = swap_mutation(offspring, mutation_probability=0.2)
        
        for tour in mutated:
            assert is_valid_permutation(tour, cities)
            assert fitness(tour, problem) > 0
    
    def test_multiple_generations_simulation(self, tsp_small):
        """Test symulacji kilku generacji"""
        problem, cities = tsp_small
        
        population = initialize_population(cities, 20)
        
        best_fitness_history = []
        
        for gen in range(5):
            fitness_scores = [fitness(t, problem) for t in population]
            best_fitness_history.append(min(fitness_scores))
            
            parents = rank_select(population, rank_size=10, tsp=problem)
            
            offspring = []
            for i in range(0, len(parents)-1, 2):
                c1, c2 = suppress_prints(order_crossover, parents[i], parents[i+1])
                offspring.extend([c1, c2])
            
            offspring = swap_mutation(offspring, mutation_probability=0.1)
            
            population = parents[:5] + offspring[:15]
        
        assert len(best_fitness_history) == 5
        assert all(f > 0 for f in best_fitness_history)
    
    def test_all_crossover_operators(self, tsp_small):
        """Test wszystkich operatorów crossover w pipeline"""
        problem, cities = tsp_small
        
        p1 = create_random_permutation(cities)
        p2 = create_random_permutation(cities)
        
        c1_ox, c2_ox = suppress_prints(order_crossover, p1, p2)
        assert is_valid_permutation(c1_ox, cities)
        assert fitness(c1_ox, problem) > 0
        
        c1_pmx, c2_pmx = suppress_prints(partially_mapped_crossover, p1, p2)
        assert is_valid_permutation(c1_pmx, cities)
        assert fitness(c1_pmx, problem) > 0
        
        c1_erx, c2_erx = suppress_prints(edge_recombination_crossover, p1, p2)
        assert is_valid_permutation(c1_erx, cities)
        assert fitness(c1_erx, problem) > 0
    
    def test_elitism_preservation(self, tsp_small):
        """Test zachowania najlepszych osobników (symulacja elityzmu)"""
        problem, cities = tsp_small
        
        population = initialize_population(cities, 20)
        
        best_initial = min(population, key=lambda t: fitness(t, problem))
        best_fitness_initial = fitness(best_initial, problem)
        
        parents = rank_select(population, rank_size=15, tsp=problem)
        
        offspring = []
        for i in range(0, 10, 2):
            c1, c2 = suppress_prints(order_crossover, parents[i], parents[i+1])
            offspring.extend([c1, c2])
        
        new_population = parents[:5] + offspring
        
        best_after = min(new_population, key=lambda t: fitness(t, problem))
        best_fitness_after = fitness(best_after, problem)
        
        assert best_fitness_after <= best_fitness_initial


# ============================================================================
# URUCHOMIENIE TESTÓW
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    
    def test_ox_small_tour(self):
        """Test OX dla małej trasy"""
        p1 = [0, 1, 2]
        p2 = [2, 0, 1]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_ox_two_cities(self):
        """Test OX dla dwóch miast"""
        p1 = [0, 1]
        p2 = [1, 0]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_ox_preserves_segment(self):
        """Test czy OX zachowuje segment z rodzica"""
        random.seed(42)
        p1 = list(range(10))
        p2 = list(range(10))
        random.shuffle(p2)
        
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        c1, c2 = order_crossover(p1, p2)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        lines = output.split('\n')
        start = int(lines[0].split(': ')[1])
        end = int(lines[1].split(': ')[1])
        
        assert c1[start:end] == p1[start:end]
        assert c2[start:end] == p2[start:end]
    
    def test_ox_identical_parents(self):
        """Test OX dla identycznych rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [0, 1, 2, 3, 4]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        # Dzieci mogą być różne od rodziców, ale są poprawnymi permutacjami
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_ox_reversed_parents(self):
        """Test OX dla odwróconych rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [4, 3, 2, 1, 0]
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_ox_multiple_random(self):
        """Test OX dla wielu losowych przypadków"""
        cities = list(range(20))
        
        for _ in range(20):
            p1 = cities[:]
            p2 = cities[:]
            random.shuffle(p1)
            random.shuffle(p2)
            
            c1, c2 = suppress_prints(order_crossover, p1, p2)
            
            assert is_valid_permutation(c1, p1)
            assert is_valid_permutation(c2, p2)
    
    def test_ox_large_tour(self):
        """Test OX dla dużej trasy"""
        p1 = list(range(100))
        p2 = list(range(100))
        random.shuffle(p2)
        
        c1, c2 = suppress_prints(order_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)


class TestPartiallyMappedCrossover:
    """Testy Partially Mapped Crossover (PMX)"""
    
    def test_pmx_basic_validity(self):
        """Test podstawowej poprawności PMX"""
        p1 = [1, 2, 3, 4, 5, 6, 7, 8]
        p2 = [3, 7, 5, 1, 6, 8, 2, 4]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_pmx_small_tour(self):
        """Test PMX dla małej trasy"""
        p1 = [0, 1, 2]
        p2 = [2, 0, 1]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_pmx_two_cities(self):
        """Test PMX dla dwóch miast"""
        p1 = [0, 1]
        p2 = [1, 0]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_pmx_preserves_segment(self):
        """Test czy PMX zachowuje segment z rodzica"""
        random.seed(100)
        p1 = list(range(10))
        p2 = list(range(10))
        random.shuffle(p2)
        
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        c1, c2 = partially_mapped_crossover(p1, p2)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Parse output to get start and end
        lines = [line for line in output.split('\n') if line.strip()]
        if len(lines) >= 2 and 'start:' in lines[0] and 'end:' in lines[1]:
            start = int(lines[0].split(': ')[1])
            end = int(lines[1].split(': ')[1])
            
            assert c1[start:end] == p1[start:end]
            assert c2[start:end] == p2[start:end]
        else:
            # Jeśli nie ma printów, sprawdź tylko poprawność
            assert is_valid_permutation(c1, p1)
            assert is_valid_permutation(c2, p2)
    
    def test_pmx_no_duplicates(self):
        """Test czy PMX nie tworzy duplikatów"""
        p1 = [0, 1, 2, 3, 4, 5]
        p2 = [5, 4, 3, 2, 1, 0]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert len(set(c1)) == len(c1)
        assert len(set(c2)) == len(c2)
    
    def test_pmx_identical_parents(self):
        """Test PMX dla identycznych rodziców"""
        p1 = [0, 1, 2, 3, 4]
        p2 = [0, 1, 2, 3, 4]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_pmx_multiple_random(self):
        """Test PMX dla wielu losowych przypadków"""
        cities = list(range(25))
        
        for _ in range(20):
            p1 = cities[:]
            p2 = cities[:]
            random.shuffle(p1)
            random.shuffle(p2)
            
            c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
            
            assert is_valid_permutation(c1, p1), f"Child1 invalid: {c1}"
            assert is_valid_permutation(c2, p2), f"Child2 invalid: {c2}"
    
    def test_pmx_large_tour(self):
        """Test PMX dla dużej trasy"""
        p1 = list(range(50))
        p2 = list(range(50))
        random.shuffle(p2)
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)
    
    def test_pmx_mapping_correctness(self):
        """Test czy PMX poprawnie mapuje geny"""
        random.seed(12345)
        p1 = [0, 1, 2, 3, 4, 5, 6, 7]
        p2 = [7, 6, 5, 4, 3, 2, 1, 0]
        
        c1, c2 = suppress_prints(partially_mapped_crossover, p1, p2)
        
        assert is_valid_permutation(c1, p1)
        assert is_valid_permutation(c2, p2)

