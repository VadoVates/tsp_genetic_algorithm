import matplotlib.pyplot as plt
from tsp_problem import TSPProblem


def plot_tour(tsp: TSPProblem, tour: list[int], title: str = "Trasa TSP"):
    """
    Rysuje mapę trasy TSP
    
    Args:
        tsp: Obiekt TSPProblem z współrzędnymi miast
        tour: Lista miast w kolejności odwiedzania
        title: Tytuł wykresu
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Pobierz współrzędne dla trasy
    x_coords = [tsp.coordinates[city][0] for city in tour]
    y_coords = [tsp.coordinates[city][1] for city in tour]
    
    # Dodaj pierwszy punkt na końcu aby zamknąć cykl
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    # Rysuj ścieżkę
    ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, alpha=0.6, label='Trasa')
    
    # Rysuj punkty (miasta)
    ax.scatter(x_coords[:-1], y_coords[:-1], c='red', s=100, zorder=5, 
              edgecolors='black', linewidth=1.5, label='Miasta')
    
    # Zaznacz punkt startowy
    ax.scatter(x_coords[0], y_coords[0], c='green', s=200, zorder=6, 
              marker='*', edgecolors='black', linewidth=2, label='Start')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    return fig


def plot_convergence(history: list[float], title: str = "Zbieżność algorytmu"):
    """
    Rysuje wykres zbieżności algorytmu genetycznego
    
    Args:
        history: Lista wartości fitness dla każdej generacji
        title: Tytuł wykresu
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    generations = list(range(1, len(history) + 1))
    
    # Wykres liniowy
    ax.plot(generations, history, 'b-', linewidth=2, label='Najlepsza odległość')
    
    # Wypełnienie pod wykresem
    ax.fill_between(generations, history, alpha=0.3)
    
    # Zaznacz najlepszy wynik
    best_gen = history.index(min(history)) + 1
    best_value = min(history)
    ax.scatter([best_gen], [best_value], c='red', s=150, zorder=5, 
              edgecolors='black', linewidth=2, label=f'Minimum: {best_value:.2f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Generacja')
    ax.set_ylabel('Najlepsza odległość')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Dodaj informacje o poprawie
    if len(history) > 1:
        improvement = ((history[0] - history[-1]) / history[0] * 100)
        ax.text(0.02, 0.98, f'Poprawa: {improvement:.2f}%', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
    
    return fig


def plot_comparison(histories: dict[str, list[float]], title: str = "Porównanie metod"):
    """
    Rysuje porównanie zbieżności dla różnych konfiguracji
    
    Args:
        histories: Słownik {nazwa_metody: lista_wartości_fitness}
        title: Tytuł wykresu
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors
    
    for idx, (method_name, history) in enumerate(histories.items()):
        generations = list(range(1, len(history) + 1))
        color = colors[idx % len(colors)]
        
        ax.plot(generations, history, linewidth=2, label=method_name, 
               color=color, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Generacja')
    ax.set_ylabel('Najlepsza odległość')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    return fig


def plot_statistics(history: list[float], window_size: int = 50):
    """
    Rysuje statystyki zbieżności z ruchomą średnią
    
    Args:
        history: Lista wartości fitness
        window_size: Rozmiar okna dla ruchomej średniej
        
    Returns:
        Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    generations = list(range(1, len(history) + 1))
    
    # Górny wykres - wartości z ruchomą średnią
    ax1.plot(generations, history, 'b-', linewidth=1, alpha=0.5, label='Wartości')
    
    if len(history) >= window_size:
        moving_avg = []
        for i in range(len(history)):
            start = max(0, i - window_size + 1)
            moving_avg.append(sum(history[start:i+1]) / (i - start + 1))
        
        ax1.plot(generations, moving_avg, 'r-', linewidth=2, 
                label=f'Średnia krocząca ({window_size})')
    
    ax1.set_title('Zbieżność z średnią kroczącą', fontweight='bold')
    ax1.set_xlabel('Generacja')
    ax1.set_ylabel('Odległość')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Dolny wykres - poprawa między generacjami
    improvements = [0]
    for i in range(1, len(history)):
        improvement = history[i-1] - history[i]
        improvements.append(improvement)
    
    ax2.bar(generations, improvements, color='green', alpha=0.6, 
           label='Poprawa', edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax2.set_title('Poprawa między generacjami', fontweight='bold')
    ax2.set_xlabel('Generacja')
    ax2.set_ylabel('Zmiana odległości')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    return fig