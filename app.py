"""
app.py – Aplikacja Streamlit do wizualizacji algorytmu genetycznego dla TSP
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tsp_problem
from tsp_problem import TSPProblem
from visualization import plot_statistics, plot_tour, plot_convergence
from genetic_algorithm import run_genetic_algorithm, RESULTS_FILE

DATASETS = {
    "ATT48": "data/att48.tsp",
    "Berlin52": "data/berlin52.tsp"
}

SOLUTION_PATHS = {
    "ATT48": "data/att48.opt.tour",
    "Berlin52": "data/berlin52.opt.tour"
}

OPTIMAL_SOLUTIONS = {
    "ATT48": tsp_problem.optimal_tour(SOLUTION_PATHS["ATT48"]),
    "Berlin52": tsp_problem.optimal_tour(SOLUTION_PATHS["Berlin52"])
}

# Konfiguracja strony
st.set_page_config(
    page_title="TSP Genetic Algorithm",
    page_icon="🧬",
    layout="wide"
)

# Styl CSS
# Styl CSS
st.markdown("""
<style>

/* ===== METRYKI ===== */
.metric-card {
    background-color: red;
    padding: 15px;
    border-radius: 10px;
    margin: 5px 0;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div > div {
    background-color: #00cc00;
}

.stSidebar {
    padding: 5px !important;
    width: 24rem;
}
section[data-testid="stSidebar"] > div {
    padding-top: 5px !important;
    padding-bottom: 5px !important;
}

div[data-testid="stSidebarHeader"] {
    height: 0rem !important;
}

hr {
    margin: 1rem 0rem !important;
}
section.main > div {
    max-width: 1000px;
    margin-left: auto;
    margin-right: auto;
}
.stMainBlockContainer{
    max-width: 1400px;
    margin-left: auto;
    margin-right: auto;
}

</style>
""", unsafe_allow_html=True)


def load_experiments() -> pd.DataFrame:
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame()

# noinspection PyShadowingNames
def create_visualisation_callback(tsp: TSPProblem, map_placeholder, chart_placeholder, metrics_placeholder,
                                  progress_bar, status_text):
    def on_generation(generation, generations_total, tour, distance, hist, elapsed_time,
                      initial_distance, opt_distance):
        with map_placeholder.container():
            fig_map = plot_tour(tsp, tour, f"Najlepsza trasa (Gen {generation + 1})")
            st.pyplot(fig_map)
            plt.close(fig_map)

        # Wykres zbieżności
        with chart_placeholder.container():
            fig_chart = plot_convergence(hist, "Zbieżność algorytmu")
            st.pyplot(fig_chart)
            plt.close(fig_chart)

        # Metryki
        improv = ((initial_distance - distance) / initial_distance * 100) if initial_distance else 0
        diff_from_optimal = distance - opt_distance if opt_distance else None

        with metrics_placeholder.container():
            column1, column2, column3 = st.columns(3)
            with column1:
                st.metric("Najlepsza odległość", f"{distance:.2f}")
                st.metric("Generacja", f"{generation + 1}/{generations_total}")
            with column2:
                if opt_distance and diff_from_optimal is not None:
                    st.metric("Różnica od optimum", f"{diff_from_optimal:.2f}",
                      delta=f"{(diff_from_optimal / opt_distance * 100):.2f}%", delta_color="inverse")
                st.metric("Czas wykonania", f"{elapsed_time:.2f}s")
            with column3:
                st.metric("Poprawa", f"{improv:.2f}%", delta=f"{improv:.1f}%")
                if opt_distance:
                    st.metric("Optimum", f"{opt_distance}")

        # Progress bar
        progress = (generation + 1) / generations_total
        progress_bar.progress(progress)
        status_text.text(f"Generacja {generation + 1}/{generations_total} - Dystans: {distance:.2f}")

    return on_generation

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
population_size = st.sidebar.slider("Rozmiar populacji", 50, 500, 300, 10)
generations = st.sidebar.slider("Liczba generacji", 100, 2000, 600, 50)
elitism_percent = st.sidebar.slider("Rozmiar elity (%)", 0, 20, 10, 1)
elitism_count = int(population_size * elitism_percent / 100)

# Parametry operatorów
st.sidebar.subheader("Operatory genetyczne")
mutation_prob = st.sidebar.slider("Prawdopodobieństwo mutacji", 0.01, 0.2, 0.1, 0.01)
crossover_prob = st.sidebar.slider("Prawdopodobieństwo krzyżowania", 0.5, 1.0, 1.0, 0.05)

# Metody
st.sidebar.subheader("Metody")
selection_method = st.sidebar.selectbox(
    "Metoda selekcji",
    options=["Rank Selection", "Tournament Selection", "Roulette Selection"]
)

crossover_method = st.sidebar.selectbox(
    "Metoda krzyżowania",
    options=["Edge Recombination (ERX)", "Order Crossover (OX)", "Partially Mapped Crossover (PMX)"]
)

mutation_method = st.sidebar.selectbox(
    "Metoda mutacji",
    options=["Inversion Mutation", "Swap Mutation", "Scramble Mutation"]
)

# Parametry dodatkowe dla metod selekcji
st.sidebar.subheader("Parametry selekcji")
if selection_method == "Rank Selection":
    rank_size = st.sidebar.slider(label="Rank size", min_value=10, max_value=population_size,
                                  value=min(200, population_size), step=5)
    roulette_size = 0
    tournament_size = 0
elif selection_method == "Tournament Selection":
    tournament_size = st.sidebar.slider(label="Tournament size", min_value=2, max_value=10, value=3, step=1)
    rank_size = 0
    roulette_size = 0
else:  # Roulette Selection
    roulette_size = st.sidebar.slider(label="Roulette size", min_value=10, max_value=population_size,
                                      value=min(80, population_size), step=10)
    rank_size = 0
    tournament_size = 0

st.sidebar.markdown("---")

# Przycisk START
start_button = st.sidebar.button("🚀 START", type="primary", width='stretch')

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

    if tsp.problem.edge_weight_type == "ATT":
        problem_type = "ATT48"
    elif tsp.problem.edge_weight_type == "EUC_2D":
        problem_type = "Berlin52"
    else:
        problem_type = None

    if problem_type:
        optimal_distance = tsp.tour_length(OPTIMAL_SOLUTIONS[problem_type])
    else:
        optimal_distance = None

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
        on_generation_callback = create_visualisation_callback(tsp, map_placeholder, chart_placeholder,
                                                               metrics_placeholder, progress_bar, status_text)
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
            optimal_distance=optimal_distance,
            problem_type=problem_type,
            on_generation=on_generation_callback,
            generate_csv=True
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
        if optimal_distance:
            diff = best_distance - optimal_distance
            st.metric("Różnica od optimum", f"{diff:.2f}", 
                     delta=f"{(diff/optimal_distance*100):.2f}%", delta_color="inverse")
    
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
        df = load_experiments()

        if df.empty:
            st.info ("Brak zapisanych eksperymentów. Uruchom algorytm przynajmniej raz aby zobaczyć historię")
        else:
            # filtrowanie
            datasets_in_history = df["problem"].unique().tolist()
            selected_dataset = st.selectbox("Filtruj po datasecie", ["Wszystkie"] + datasets_in_history)

            df_filtered = df if selected_dataset == "Wszystkie" else df[df["problem"] == selected_dataset]

            # najlepsze wyniki
            st.subheader("Najlepsze wyniki")
            best_per_dataset = df.loc[df.groupby("problem")["best_distance"].idxmin()]
            st.dataframe(
                best_per_dataset[["problem", "best_distance", "gap_percent", "selection_method", "crossover_method",
                                  "mutation_method", "total_time_s"]],
                width='stretch',
                hide_index=True
            )

            st.subheader("Historia eksperymentów")
            st.dataframe(
                df_filtered.sort_values("timestamp", ascending=False),
                width='stretch',
                hide_index=True
            )

            if st.button("Wyczyść historię"):
                RESULTS_FILE.unlink(missing_ok=True)
                st.rerun()
    
    # Historia trasy
    with st.expander("🔍 Zobacz szczegóły trasy"):
        st.write(f"**Kolejność miast:** {best_tour}")
        if best_tour is not None:
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