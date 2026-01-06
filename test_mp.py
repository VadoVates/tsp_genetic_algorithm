"""
tests_mp.py - Batch testing TSP GA (multiprocessing)
Uruchom: python tests_mp.py
"""

import itertools
import csv
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from tsp_problem import TSPProblem, optimal_tour
from genetic_algorithm import run_genetic_algorithm

DATASETS = {
    "ATT48": "data/att48.tsp",
    "Berlin52": "data/berlin52.tsp"
}

SOLUTION_PATHS = {
    "ATT48": "data/att48.opt.tour",
    "Berlin52": "data/berlin52.opt.tour"
}

RESULTS_FILE = Path("results/experiments.csv")
FLUSH_EVERY = 500

# -------- helpers --------

def float_range(start: float, stop: float, step: float) -> list[float]:
    result = []
    current = start
    while current <= stop + step / 2:
        result.append(round(current, 4))
        current += step
    return result

def int_range(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop + 1, step))

def _csv_append_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

# -------- configs --------

COMMON_CONFIG = {
    "population_size": int_range(50, 200, 50),       # przykładowo zawężone
    "generations": int_range(200, 1000, 200),
    "elitism_percent": float_range(0, 10, 5),
    "mutation_prob": [0.02, 0.05, 0.1],
    "crossover_prob": [0.7, 0.85, 0.95],
    "repeats": 1
}

SELECTION_CONFIGS = {
    "Rank Selection": {"rank_size": int_range(10, 50, 10)},
    "Tournament Selection": {"tournament_size": int_range(2, 8, 2)},
    "Roulette Selection": {"roulette_size": int_range(10, 50, 20)}
}

CROSSOVER_METHODS = [
    "Order Crossover (OX)",
    "Partially Mapped Crossover (PMX)",
    "Edge Recombination (ERX)"
]

MUTATION_METHODS = [
    "Swap Mutation",
    "Inversion Mutation",
    "Scramble Mutation"
]

# -------- worker globals (per process) --------

_WORK_TSP: TSPProblem | None = None
_WORK_OPTIMAL_DISTANCE: int | None = None
_WORK_DATASET: str | None = None

def _init_worker(dataset: str) -> None:
    global _WORK_TSP, _WORK_OPTIMAL_DISTANCE, _WORK_DATASET
    _WORK_DATASET = dataset
    tsp = TSPProblem(DATASETS[dataset])
    opt = optimal_tour(SOLUTION_PATHS[dataset])
    _WORK_TSP = tsp
    _WORK_OPTIMAL_DISTANCE = tsp.tour_length(opt)

def _task_run(args: tuple) -> dict:
    """
    args = (
        selection_method, selection_param_name, selection_param_value,
        crossover_method, mutation_method,
        pop_size, gens, elite_pct, mut_prob, cross_prob,
        repeat
    )
    """
    global _WORK_TSP, _WORK_OPTIMAL_DISTANCE, _WORK_DATASET
    if _WORK_TSP is None or _WORK_OPTIMAL_DISTANCE is None or _WORK_DATASET is None:
        raise RuntimeError("Worker not initialized")

    (
        selection_method, sel_param_name, sel_param_value,
        crossover_method, mutation_method,
        pop_size, gens, elite_pct, mut_prob, cross_prob,
        repeat
    ) = args

    # parametry selekcji – tylko właściwy ma znaczenie
    if selection_method == "Rank Selection":
        rank_size = min(int(sel_param_value), pop_size)
        tournament_size = 3
        roulette_size = 50
    elif selection_method == "Tournament Selection":
        rank_size = 30
        tournament_size = min(int(sel_param_value), pop_size)
        roulette_size = 50
    else:
        rank_size = 30
        tournament_size = 3
        roulette_size = min(int(sel_param_value), pop_size)

    elitism_count = int(pop_size * elite_pct / 100)

    # deterministyczny seed per test (dla porównywalności)
    seed = hash((
        _WORK_DATASET, selection_method, sel_param_name, sel_param_value,
        crossover_method, mutation_method,
        pop_size, gens, elite_pct, mut_prob, cross_prob, repeat
    )) & 0xFFFFFFFF

    start = time.time()
    best_tour, best_distance, history, total_time = run_genetic_algorithm(
        tsp=_WORK_TSP,
        population_size=pop_size,
        generations=gens,
        mutation_prob=mut_prob,
        crossover_prob=cross_prob,
        elitism_count=elitism_count,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_method=mutation_method,
        rank_size=rank_size,
        roulette_size=roulette_size,
        tournament_size=tournament_size,
        optimal_distance=_WORK_OPTIMAL_DISTANCE,
        problem_type=_WORK_DATASET,
        on_generation=None,
        seed=seed,                # dodaj obsługę w run_genetic_algorithm
    )
    elapsed = time.time() - start  # opcjonalnie

    gap_percent = (best_distance - _WORK_OPTIMAL_DISTANCE) / _WORK_OPTIMAL_DISTANCE * 100

    return {
        "timestamp": time.time(),
        "problem": _WORK_DATASET,
        "best_distance": best_distance,
        "optimal_distance": _WORK_OPTIMAL_DISTANCE,
        "gap_percent": gap_percent,
        "total_time_s": total_time,
        "wall_time_s": elapsed,
        "generations": gens,
        "population_size": pop_size,
        "mutation_prob": mut_prob,
        "crossover_prob": cross_prob,
        "elitism_percent": elite_pct,
        "selection_method": selection_method,
        "selection_param_name": sel_param_name,
        "selection_param_value": sel_param_value,
        "crossover_method": crossover_method,
        "mutation_method": mutation_method,
        "repeat": repeat,
    }

# -------- main --------

def _iter_tasks(
    selection_methods: list[str],
    crossover_methods: list[str],
    mutation_methods: list[str]
):
    for selection_method in selection_methods:
        sel_cfg = SELECTION_CONFIGS[selection_method]
        sel_param_name = next(iter(sel_cfg.keys()))
        sel_param_values = sel_cfg[sel_param_name]

        for crossover_method, mutation_method in itertools.product(crossover_methods, mutation_methods):
            for pop_size, gens, elite_pct, mut_prob, cross_prob, sel_param_value in itertools.product(
                COMMON_CONFIG["population_size"],
                COMMON_CONFIG["generations"],
                COMMON_CONFIG["elitism_percent"],
                COMMON_CONFIG["mutation_prob"],
                COMMON_CONFIG["crossover_prob"],
                sel_param_values
            ):
                for repeat in range(COMMON_CONFIG["repeats"]):
                    yield (
                        selection_method, sel_param_name, sel_param_value,
                        crossover_method, mutation_method,
                        pop_size, gens, elite_pct, mut_prob, cross_prob,
                        repeat
                    )

def _count_total(selection_methods: list[str], crossover_methods: list[str], mutation_methods: list[str]) -> int:
    base = (
        len(COMMON_CONFIG["population_size"]) *
        len(COMMON_CONFIG["generations"]) *
        len(COMMON_CONFIG["elitism_percent"]) *
        len(COMMON_CONFIG["mutation_prob"]) *
        len(COMMON_CONFIG["crossover_prob"]) *
        len(crossover_methods) *
        len(mutation_methods) *
        COMMON_CONFIG["repeats"]
    )
    total = 0
    for sel in selection_methods:
        sel_param_values = list(SELECTION_CONFIGS[sel].values())[0]
        total += base * len(sel_param_values)
    return total

def run_batch_mp(
    dataset: str,
    selection_methods: list[str],
    crossover_methods: list[str],
    mutation_methods: list[str],
    workers: int | None = None,
) -> None:
    csv_fieldnames = [
        "timestamp", "problem",
        "best_distance", "optimal_distance", "gap_percent",
        "total_time_s", "wall_time_s",
        "generations", "population_size",
        "mutation_prob", "crossover_prob", "elitism_percent",
        "selection_method", "selection_param_name", "selection_param_value",
        "crossover_method", "mutation_method",
        "repeat",
    ]

    total_tests = _count_total(selection_methods, crossover_methods, mutation_methods)
    print(f"Dataset: {dataset}")
    print(f"Łącznie testów: {total_tests}")

    buffer_rows: list[dict] = []
    best = {"gap_percent": float("inf"), "row": None}

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    done = 0
    tasks_iter = _iter_tasks(selection_methods, crossover_methods, mutation_methods)

#    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(dataset,)) as ex:
    with Pool(processes=workers, initializer=_init_worker, initargs=(dataset,)) as pool:
        # chunksize ogranicza narzut IPC; dostosuj (np. 50–500)
        for row in pool.imap_unordered(_task_run, tasks_iter, chunksize=50):
            done += 1
            buffer_rows.append(row)

            if row["gap_percent"] < best["gap_percent"]:
                best = {"gap_percent": row["gap_percent"], "row": row}

            if done % 100 == 0:
                print(f"[{done}/{total_tests}] ({100*done/total_tests:.1f}%)")

            if len(buffer_rows) >= FLUSH_EVERY:
                _csv_append_rows(RESULTS_FILE, buffer_rows, csv_fieldnames)
                buffer_rows.clear()

    if buffer_rows:
        _csv_append_rows(RESULTS_FILE, buffer_rows, csv_fieldnames)
        buffer_rows.clear()

    print("ZAKOŃCZONO")
    if best["row"] is not None:
        r = best["row"]
        print(f"BEST gap={r['gap_percent']:.2f}% dist={r['best_distance']:.0f} "
              f"sel={r['selection_method']} {r['selection_param_name']}={r['selection_param_value']} "
              f"cx={r['crossover_method']} mut={r['mutation_method']} "
              f"pop={r['population_size']} gen={r['generations']} elite={r['elitism_percent']} "
              f"pm={r['mutation_prob']} pc={r['crossover_prob']}")
    print(f"Wyniki: {RESULTS_FILE}")

if __name__ == "__main__":
    DATASET = "ATT48"

    selection_methods = ["Rank Selection", "Tournament Selection", "Roulette Selection"]
    crossover_methods = CROSSOVER_METHODS
    mutation_methods = MUTATION_METHODS

    total = _count_total(selection_methods, crossover_methods, mutation_methods)
    print(f"Zaplanowano {total} testów")
    confirm = input("Kontynuować? (t/n): ")
    if confirm.lower() == "t":
        run_batch_mp(DATASET, selection_methods, crossover_methods, mutation_methods, workers=23)
