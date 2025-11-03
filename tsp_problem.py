"""
tsp_problem.py – warstwa dostępu do TSPLIB:
ładowanie *.tsp i *.opt.tour (tsplib95), metody distance(u,v),
tour_length(route), walidacja trasy, konwersja do networkx (opcjonalnie).
"""

import tsplib95
tsp_problem = tsplib95.load_problem("data/att48.tsp")
tsp_solution = tsplib95.load_solution("data/att48.opt.tour")

berlin_problem = tsplib95.load_problem("data/berlin52.tsp")
berlin_solution = tsplib95.load_solution("berlin52.opt.tour")