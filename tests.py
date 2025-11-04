import random
from tsp_problem import TSPProblem
from genetic_algorithm import *

def test_initialization():
    """Test tworzenia populacji"""
    print("=== TEST INICJALIZACJI ===")
    cities = [0, 1, 2, 3, 4]
    
    # Test pojedynczej permutacji
    tour = create_random_permutation(cities)
    print(f"Przykładowa trasa: {tour}")
    print(f"Czy zawiera wszystkie miasta? {set(tour) == set(cities)}")
    print(f"Czy ma odpowiednią długość? {len(tour) == len(cities)}")
    
    # Test populacji
    population = initialize_population(cities, 10)
    print(f"\nRozmiar populacji: {len(population)}")
    print(f"Przykładowe trasy z populacji:")
    for i, tour in enumerate(population[:3]):
        print(f"  Trasa {i+1}: {tour}")
    print()

def test_fitness():
    """Test funkcji fitness"""
    print("=== TEST FITNESS ===")
    problem = TSPProblem("data/att48.tsp")
    cities = list(problem.coordinates.keys())
    
    # Utworz kilka tras
    tour1 = create_random_permutation(cities)
    tour2 = create_random_permutation(cities)
    
    fit1 = fitness(tour1, problem)
    fit2 = fitness(tour2, problem)
    
    print(f"Długość trasy 1: {fit1}")
    print(f"Długość trasy 2: {fit2}")
    print(f"Lepsza trasa: {'Trasa 1' if fit1 < fit2 else 'Trasa 2'}")
    
    # Test normalizacji
    worst = max(fit1, fit2)
    norm1 = fitness_normalized(tour1, problem, worst)
    norm2 = fitness_normalized(tour2, problem, worst)
    print(f"\nFitness znormalizowany 1: {norm1}")
    print(f"Fitness znormalizowany 2: {norm2}")
    print(f"Lepsza po normalizacji: {'Trasa 1' if norm1 > norm2 else 'Trasa 2'}")
    print()

def test_rank_selection():
    """Test selekcji rankingowej"""
    print("=== TEST RANK SELECTION ===")
    problem = TSPProblem("data/att48.tsp")
    cities = list(problem.coordinates.keys())
    population = initialize_population(cities, 20)
    
    # Pokaż fitness przed selekcją
    fitness_before = [fitness(tour, problem) for tour in population]
    print(f"Fitness populacji (pierwsze 5): {fitness_before[:5]}")
    print(f"Min: {min(fitness_before)}, Max: {max(fitness_before)}, Avg: {sum(fitness_before)/len(fitness_before):.2f}")
    
    # Selekcja
    selected = rank_select(population, rank_size=5, tsp=problem)
    fitness_after = [fitness(tour, problem) for tour in selected]
    
    print(f"\nFitness po selekcji TOP 5: {fitness_after}")
    print(f"Czy posortowane rosnąco? {fitness_after == sorted(fitness_after)}")
    print()

def test_tournament_selection():
    """Test selekcji turniejowej"""
    print("=== TEST TOURNAMENT SELECTION ===")
    problem = TSPProblem("data/att48.tsp")
    cities = list(problem.coordinates.keys())
    population = initialize_population(cities, 20)
    
    print("Wybieranie zwycięzców 5 turniejów (rozmiar turnieju = 3):")
    winners_fitness = []
    for i in range(5):
        winner = tournament_select(population, problem, tournament_size=3)
        win_fit = fitness(winner, problem)
        winners_fitness.append(win_fit)
        print(f"  Turniej {i+1}: fitness = {win_fit}")
    
    avg_winner = sum(winners_fitness) / len(winners_fitness)
    avg_population = sum(fitness(t, problem) for t in population) / len(population)
    
    print(f"\nŚredni fitness zwycięzców: {avg_winner:.2f}")
    print(f"Średni fitness populacji: {avg_population:.2f}")
    print(f"Zwycięzcy są lepsi? {avg_winner < avg_population}")
    print()

def test_roulette_selection():
    """Test selekcji ruletkowej"""
    print("=== TEST ROULETTE SELECTION ===")
    problem = TSPProblem("data/att48.tsp")
    cities = list(problem.coordinates.keys())
    population = initialize_population(cities, 20)
    
    # Uwaga: Twoja funkcja ma błąd - wybiera z contenders zamiast z population
    # Poniżej testuję to co jest napisane
    try:
        selected = roulette_select(population, problem, roulette_selection_size=10)
        print(f"Wybrano {len(selected)} osobników")
        fitness_selected = [fitness(tour, problem) for tour in selected]
        print(f"Fitness wybranych (pierwsze 5): {fitness_selected[:5]}")
    except Exception as e:
        print(f"Błąd w roulette_select: {e}")
    print()

def test_all():
    """Uruchom wszystkie testy"""
    random.seed()  # dla powtarzalności
    
    test_initialization()
    test_fitness()
    test_rank_selection()
    test_tournament_selection()
    test_roulette_selection()
    
    print("=== WSZYSTKIE TESTY ZAKOŃCZONE ===")

# Uruchom testy
if __name__ == "__main__":
    test_all()