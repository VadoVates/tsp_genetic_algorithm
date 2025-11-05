import random
from tsp_problem import TSPProblem
from genetic_algorithm import *
from collections import Counter

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

def is_permutation(t, base):
    return len(t) == len(base) and set(t) == set(base)

def test_order_crossover():
    p1 = [0,1,2,3,4,5,6,7,8,9]
    p2 = [3,7,5,1,9,0,2,8,6,4]
    c1, c2 = order_crossover(p1, p2)
    print(f"Rodzic 1: {p1}")
    print(f"Rodzic 2: {p2}")
    print(f"Dziecko 1: {c1}")
    print(f"Dziecko 2: {c2}")
    assert is_permutation(c1, p1)
    assert is_permutation(c2, p1)

def test_partially_mapped_crossover():
    """Test Partially Mapped Crossover"""
    print("="*60)
    print("=== TEST PARTIALLY MAPPED CROSSOVER (PMX) ===")
    print("="*60)
    
    # Test 1: Prosty przykład z dokumentacji
    print("\n--- TEST 1: Klasyczny przykład PMX ---")
    random.seed(42)
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
    parent2 = [3, 7, 5, 1, 6, 8, 2, 4]
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print()
    
    child1, child2 = partially_mapped_crossover(parent1, parent2)
    
    print(f"\nChild 1:  {child1}")
    print(f"Child 2:  {child2}")
    
    # Walidacja
    print("\n✓ Walidacja Child 1:")
    print(f"  Długość: {len(child1)} (oczekiwane: {len(parent1)})")
    print(f"  Wszystkie unikalne: {len(set(child1)) == len(child1)}")
    print(f"  Te same geny co rodzice: {set(child1) == set(parent1)}")
    print(f"  Counter: {Counter(child1)}")
    
    print("\n✓ Walidacja Child 2:")
    print(f"  Długość: {len(child2)} (oczekiwane: {len(parent2)})")
    print(f"  Wszystkie unikalne: {len(set(child2)) == len(child2)}")
    print(f"  Te same geny co rodzice: {set(child2) == set(parent2)}")
    print(f"  Counter: {Counter(child2)}")
    
    # Test 2: Mały rozmiar
    print("\n" + "="*60)
    print("--- TEST 2: Małe miasta (5 miast) ---")
    random.seed(100)
    parent1 = [0, 1, 2, 3, 4]
    parent2 = [2, 4, 1, 0, 3]
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print()
    
    child1, child2 = partially_mapped_crossover(parent1, parent2)
    
    print(f"\nChild 1:  {child1}")
    print(f"Child 2:  {child2}")
    
    assert_valid_tour(child1, parent1, "Child 1 (Test 2)")
    assert_valid_tour(child2, parent2, "Child 2 (Test 2)")
    
    # Test 3: Większy problem
    print("\n" + "="*60)
    print("--- TEST 3: Średnie miasta (15 miast) ---")
    random.seed(200)
    parent1 = list(range(15))
    parent2 = list(range(15))
    random.shuffle(parent2)
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print()
    
    child1, child2 = partially_mapped_crossover(parent1, parent2)
    
    print(f"\nChild 1:  {child1}")
    print(f"Child 2:  {child2}")
    
    assert_valid_tour(child1, parent1, "Child 1 (Test 3)")
    assert_valid_tour(child2, parent2, "Child 2 (Test 3)")
    
    # Test 4: Wiele losowych krzyżowań
    print("\n" + "="*60)
    print("--- TEST 4: 10 losowych krzyżowań (sprawdzenie stabilności) ---")
    cities = list(range(20))
    all_valid = True
    
    for i in range(10):
        p1 = cities[:]
        p2 = cities[:]
        random.shuffle(p1)
        random.shuffle(p2)
        
        # Wyłącz printy dla tego testu
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            c1, c2 = partially_mapped_crossover(p1, p2)
            sys.stdout = old_stdout
            
            # Sprawdź poprawność
            valid1 = (len(c1) == len(p1) and 
                     len(set(c1)) == len(c1) and 
                     set(c1) == set(p1))
            valid2 = (len(c2) == len(p2) and 
                     len(set(c2)) == len(c2) and 
                     set(c2) == set(p2))
            
            if valid1 and valid2:
                print(f"  Krzyżowanie {i+1}: ✓ OK")
            else:
                print(f"  Krzyżowanie {i+1}: ✗ BŁĄD!")
                all_valid = False
                
        except Exception as e:
            sys.stdout = old_stdout
            print(f"  Krzyżowanie {i+1}: ✗ WYJĄTEK: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✓ Wszystkie 10 krzyżowań zakończone sukcesem!")
    else:
        print("\n✗ Niektóre krzyżowania zakończyły się błędem!")
    
    # Test 5: Dziedziczenie materiału genetycznego
    print("\n" + "="*60)
    print("--- TEST 5: Analiza dziedziczenia ---")
    random.seed(300)
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parent2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    # Wyłącz printy
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    child1, child2 = partially_mapped_crossover(parent1, parent2)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Wyciągnij start i end z outputu
    lines = output.split('\n')
    start = int(lines[0].split(': ')[1])
    end = int(lines[1].split(': ')[1])
    
    print(f"\nWylosowany segment: [{start}:{end}]")
    print(f"Child 1:  {child1}")
    print(f"  Segment z Parent1: {child1[start:end]} == {parent1[start:end]} ? {child1[start:end] == parent1[start:end]}")
    print(f"Child 2:  {child2}")
    print(f"  Segment z Parent2: {child2[start:end]} == {parent2[start:end]} ? {child2[start:end] == parent2[start:end]}")
    
    # Test 6: Przypadki brzegowe
    print("\n" + "="*60)
    print("--- TEST 6: Przypadki brzegowe ---")
    
    # Minimalna trasa (3 miasta)
    print("\nPrzypadek A: 3 miasta")
    parent1 = [0, 1, 2]
    parent2 = [2, 0, 1]
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    child1, child2 = partially_mapped_crossover(parent1, parent2)
    sys.stdout = old_stdout
    
    print(f"Child 1:  {child1}")
    print(f"Child 2:  {child2}")
    assert_valid_tour(child1, parent1, "Child 1 (3 miasta)")
    assert_valid_tour(child2, parent2, "Child 2 (3 miasta)")
    
    print("\n" + "="*60)
    print("✓ WSZYSTKIE TESTY PMX ZAKOŃCZONE!")
    print("="*60)


def assert_valid_tour(child: list[int], parent: list[int], label: str):
    """Sprawdza czy dziecko jest poprawną trasą"""
    errors = []
    
    if len(child) != len(parent):
        errors.append(f"Zła długość: {len(child)} != {len(parent)}")
    
    if len(set(child)) != len(child):
        duplicates = [x for x in child if child.count(x) > 1]
        errors.append(f"Duplikaty: {set(duplicates)}")
    
    if set(child) != set(parent):
        missing = set(parent) - set(child)
        extra = set(child) - set(parent)
        if missing:
            errors.append(f"Brakuje: {missing}")
        if extra:
            errors.append(f"Nadmiarowe: {extra}")
    
    if errors:
        print(f"\n✗ {label} - BŁĘDY:")
        for error in errors:
            print(f"    {error}")
    else:
        print(f"  ✓ {label} - poprawna trasa")    

def test_all():
    """Uruchom wszystkie testy"""    
    test_initialization()
    test_fitness()
    test_rank_selection()
    test_tournament_selection()
    test_roulette_selection()
    test_order_crossover()
    test_partially_mapped_crossover()
    
    print("=== WSZYSTKIE TESTY ZAKOŃCZONE ===")

# Uruchom testy
if __name__ == "__main__":
    random.seed()
    test_partially_mapped_crossover()
    # test_all()
    #test_order_crossover()
