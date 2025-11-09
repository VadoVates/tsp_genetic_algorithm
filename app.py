"""
app.py - Aplikacja Streamlit do wizualizacji algorytmu genetycznego dla TSP
"""

import streamlit as st
import matplotlib.pyplot as plt
import time
import random
from tsp_problem import TSPProblem
from genetic_algorithm import (
    initialize_population, fitness, rank_select, tournament_select, 
    roulette_select, order_crossover, partially_mapped_crossover,
    edge_recombination_crossover, swap_mutation, inversion_mutation,
    scramble_mutation
)
from visualization import plot_tour, plot_convergence, plot_comparison, plot_statistics

# Konfiguracja strony
st.set_page_config(
    page_title="TSP Genetic Algorithm",
    page_icon="🧬",
    layout="wide"
)

# Styl CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Optima dla datasetów
OPTIMAL_SOLUTIONS = {
    "ATT48": 10628,
    "Berlin52": 7542
}

DATASETS = {
    "ATT48": "data/att48.tsp",
    "Berlin52": "data/berlin52.tsp"
}

def run_genetic_algorithm(
    tsp: TSPProblem,
    population_size: int,
    generations: int,
    mutation_prob: float,
    crossover_prob: float,
    elitism_count: int,
    selection_method: str,
    crossover_method: str,
    mutation_method: str,
    rank_size: int,
    roulette_size: int,
    tournament_size: int,
    map_placeholder,
    chart_placeholder,
    metrics_placeholder,
    progress_bar,
    status_text
):
    """Główna pętla algorytmu genetycznego z wizualizacją na żywo"""
    
    cities = list(tsp.coordinates.keys())
    population = initialize_population(cities, population_size)
    
    history = []
    best_tour = None
    best_distance = float('inf')
    start_time = time.time()
    initial_distance = None
    
    # Metody krzyżowania
    crossover_methods = {
        "Order Crossover (OX)": order_crossover,
        "Partially Mapped Crossover (PMX)": partially_mapped_crossover,
        "Edge Recombination (ERX)": edge_recombination_crossover
    }
    
    # Metody mutacji
    mutation_methods = {
        "Swap Mutation": swap_mutation,
        "Inversion Mutation": inversion_mutation,
        "Scramble Mutation": scramble_mutation
    }
    
    crossover_func = crossover_methods[crossover_method]
    mutation_func = mutation_methods[mutation_method]
    
    for generation in range(generations):
        # Ocena fitness
        population_with_fitness = [(tour, fitness(tour, tsp)) for tour in population]
        population_with_fitness.sort(key=lambda x: x[1])
        
        current_best_tour, current_best_distance = population_with_fitness[0]
        
        if generation == 0:
            initial_distance = current_best_distance
        
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_tour = current_best_tour[:]
        
        history.append(best_distance)
        
        # Aktualizacja wizualizacji co 10 generacji lub na końcu
        if generation % 10 == 0 or generation == generations - 1:
            # Mapa trasy
            with map_placeholder.container():
                fig_map = plot_tour(tsp, best_tour, f"Najlepsza trasa (Gen {generation + 1})")
                st.pyplot(fig_map)
                plt.close(fig_map)
            
            # Wykres zbieżności
            with chart_placeholder.container():
                fig_chart = plot_convergence(history, "Zbieżność algorytmu")
                st.pyplot(fig_chart)
                plt.close(fig_chart)
            
            # Metryki
            elapsed_time = time.time() - start_time
            improvement = ((initial_distance - best_distance) / initial_distance * 100) if initial_distance else 0
            optimal = OPTIMAL_SOLUTIONS.get(st.session_state.dataset, None)
            diff_from_optimal = best_distance - optimal if optimal else None
            
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Najlepsza odległość", f"{best_distance:.2f}")
                    st.metric("Generacja", f"{generation + 1}/{generations}")
                with col2:
                    if optimal:
                        st.metric("Różnica od optimum", f"{diff_from_optimal:.2f}", 
                                 delta=f"{(diff_from_optimal/optimal*100):.2f}%", delta_color="inverse")
                    st.metric("Czas wykonania", f"{elapsed_time:.2f}s")
                with col3:
                    st.metric("Poprawa", f"{improvement:.2f}%", delta=f"{improvement:.1f}%")
                    if optimal:
                        st.metric("Optimum", f"{optimal}")
            
            # Progress bar
            progress = (generation + 1) / generations
            progress_bar.progress(progress)
            status_text.text(f"Generacja {generation + 1}/{generations} - Dystans: {best_distance:.2f}")
        
        # Elityzm
        elites = [tour for tour, _ in population_with_fitness[:elitism_count]]
        
        # Selekcja
        selected_population = [tour for tour, _ in population_with_fitness]
        
        if selection_method == "Rank Selection":
            selected = rank_select(selected_population, len(selected_population), tsp)
        elif selection_method == "Tournament Selection":
            selected = tournament_select(selected_population, tsp, tournament_size, len(selected_population))
        else:  # Roulette
            selected = roulette_select(selected_population, tsp, len(selected_population))
        
        # Krzyżowanie
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if random.random() < crossover_prob:
                child1, child2 = crossover_func(selected[i], selected[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i], selected[i + 1]])
        
        # Mutacja
        offspring = [mutation_func(tour, mutation_prob) for tour in offspring]
        
        # Nowa populacja
        population = elites + offspring[:population_size - elitism_count]
    
    total_time = time.time() - start_time
    
    return best_tour, best_distance, history, total_time

# SIDEBAR
st.sidebar.title("⚙️ Parametry algorytmu")

# Dataset
dataset = st.sidebar.selectbox(
    "Dataset",
    options=list(DATASETS.keys()),
    index=0
)

st.sidebar.markdown("---")

# Parametry populacji
st.sidebar.subheader("Populacja")
population_size = st.sidebar.slider("Rozmiar populacji", 50, 500, 100, 10)
generations = st.sidebar.slider("Liczba generacji", 100, 2000, 500, 50)
elitism_percent = st.sidebar.slider("Rozmiar elity (%)", 0, 20, 10, 1)
elitism_count = int(population_size * elitism_percent / 100)

st.sidebar.markdown("---")

# Parametry operatorów
st.sidebar.subheader("Operatory genetyczne")
mutation_prob = st.sidebar.slider("Prawdopodobieństwo mutacji", 0.01, 0.2, 0.1, 0.01)
crossover_prob = st.sidebar.slider("Prawdopodobieństwo krzyżowania", 0.5, 1.0, 0.8, 0.05)

st.sidebar.markdown("---")

# Metody
st.sidebar.subheader("Metody")
selection_method = st.sidebar.selectbox(
    "Metoda selekcji",
    options=["Rank Selection", "Tournament Selection", "Roulette Selection"]
)

crossover_method = st.sidebar.selectbox(
    "Metoda krzyżowania",
    options=["Order Crossover (OX)", "Partially Mapped Crossover (PMX)", "Edge Recombination (ERX)"]
)

mutation_method = st.sidebar.selectbox(
    "Metoda mutacji",
    options=["Swap Mutation", "Inversion Mutation", "Scramble Mutation"]
)

st.sidebar.markdown("---")

# Parametry dodatkowe dla metod selekcji
st.sidebar.subheader("Parametry selekcji")
rank_size = st.sidebar.slider("Rank size", 10, population_size, min(30, population_size), 5)
roulette_size = st.sidebar.slider("Roulette size", 10, population_size, min(80, population_size), 5)
tournament_size = st.sidebar.slider("Tournament size", 2, 10, 3, 1)

st.sidebar.markdown("---")

# Przycisk START
start_button = st.sidebar.button("🚀 START", type="primary", use_container_width=True)

# MAIN AREA
st.title("🧬 Algorytm Genetyczny dla TSP")
st.markdown(f"**Dataset:** {dataset} | **Optimum:** {OPTIMAL_SOLUTIONS.get(dataset, 'Unknown')}")

# Inicjalizacja session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = dataset
    
if 'history_log' not in st.session_state:
    st.session_state.history_log = {}

if start_button:
    st.session_state.dataset = dataset
    
    # Ładowanie problemu
    try:
        tsp = TSPProblem(DATASETS[dataset])
    except Exception as e:
        st.error(f"Błąd ładowania datasetu: {e}")
        st.stop()
    
    # Kontenery na wizualizacje
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Najlepsza trasa")
        map_placeholder = st.empty()
    
    with col2:
        st.subheader("📈 Zbieżność algorytmu")
        chart_placeholder = st.empty()
    
    st.markdown("---")
    
    # Metryki
    st.subheader("📊 Metryki w czasie rzeczywistym")
    metrics_placeholder = st.empty()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Uruchomienie algorytmu
    with st.spinner("Optymalizacja w toku..."):
        best_tour, best_distance, history, total_time = run_genetic_algorithm(
            tsp=tsp,
            population_size=population_size,
            generations=generations,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            elitism_count=elitism_count,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            rank_size=rank_size,
            roulette_size=roulette_size,
            tournament_size=tournament_size,
            map_placeholder=map_placeholder,
            chart_placeholder=chart_placeholder,
            metrics_placeholder=metrics_placeholder,
            progress_bar=progress_bar,
            status_text=status_text
        )
    
    # Zapisz do historii porównań
    config_name = f"{selection_method[:4]}-{crossover_method[:2]}-{mutation_method[:4]}"
    st.session_state.history_log[config_name] = history
    
    # Podsumowanie końcowe
    st.markdown("---")
    st.success("✅ Optymalizacja zakończona!")
    
    st.subheader("📋 Podsumowanie końcowe")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Najlepsza odległość", f"{best_distance:.2f}")
    
    with col2:
        optimal = OPTIMAL_SOLUTIONS.get(dataset, None)
        if optimal:
            diff = best_distance - optimal
            st.metric("Różnica od optimum", f"{diff:.2f}", 
                     delta=f"{(diff/optimal*100):.2f}%", delta_color="inverse")
    
    with col3:
        st.metric("Całkowity czas", f"{total_time:.2f}s")
    
    with col4:
        improvement = ((history[0] - best_distance) / history[0] * 100)
        st.metric("Całkowita poprawa", f"{improvement:.2f}%")
    
    # Dodatkowe wykresy statystyczne
    st.markdown("---")
    st.subheader("📊 Analiza szczegółowa")
    
    tab1, tab2 = st.tabs(["Statystyki zbieżności", "Historia porównań"])
    
    with tab1:
        fig_stats = plot_statistics(history, window_size=50)
        st.pyplot(fig_stats)
        plt.close(fig_stats)
    
    with tab2:
        if len(st.session_state.history_log) > 1:
            fig_comp = plot_comparison(st.session_state.history_log, 
                                      "Porównanie różnych konfiguracji")
            st.pyplot(fig_comp)
            plt.close(fig_comp)
            
            if st.button("🗑️ Wyczyść historię porównań"):
                st.session_state.history_log = {}
                st.rerun()
        else:
            st.info("Uruchom algorytm z różnymi konfiguracjami aby zobaczyć porównanie")
    
    # Historia trasy
    with st.expander("🔍 Zobacz szczegóły trasy"):
        st.write(f"**Kolejność miast:** {best_tour}")
        st.write(f"**Liczba miast:** {len(best_tour)}")
        st.write(f"**Początkowa odległość:** {history[0]:.2f}")
        st.write(f"**Końcowa odległość:** {best_distance:.2f}")
    
    status_text.text("✅ Gotowe!")
    
else:
    # Ekran powitalny
    st.info("👈 Ustaw parametry w panelu bocznym i kliknij START aby rozpocząć optymalizację")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Instrukcja
        1. Wybierz dataset (ATT48 lub Berlin52)
        2. Dostosuj parametry algorytmu
        3. Wybierz metody selekcji, krzyżowania i mutacji
        4. Kliknij START
        5. Obserwuj optymalizację na żywo
        """)
    
    with col2:
        st.markdown("""
        ### ℹ️ Informacje
        - **ATT48**: 48 miast, optimum = 10628
        - **Berlin52**: 52 miasta, optimum = 7542
        - Algorytm aktualizuje wizualizacje co 10 generacji
        - Po zakończeniu zobaczysz pełne podsumowanie
        - Możesz porównać różne konfiguracje w zakładce "Historia porównań"
        """)