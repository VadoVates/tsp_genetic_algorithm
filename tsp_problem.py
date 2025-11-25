"""
tsp_problem.py – warstwa dostępu do TSPLIB:
ładowanie *.tsp i *.opt.tour (tsplib95), metody distance(u,v),
tour_length(route), walidacja trasy, konwersja do networkx (opcjonalnie).
"""

import tsplib95

# nearest integer
def nint(result:float) -> int:
    return (int(result + 0.5))

# algorytm att
def att (x1:int, y1:int, x2:int, y2:int) -> int:
    xd = x1 - x2
    yd = y1 - y2
    rij = ((xd*xd + yd*yd) / 10) ** 0.5
    tij = nint (rij)
    dij = tij + 1 if tij < rij else tij
    return dij

# odległość euklidesowa
def euc_2d (x1: int, y1: int, x2: int, y2: int) -> int:
    xd = x1 - x2
    yd = y1 - y2
    dij = nint ((xd*xd + yd*yd) ** 0.5)
    return dij

class TSPProblem:
    def __init__(self, problem_path = "data/att48.tsp") -> None:
        self.problem = tsplib95.load(problem_path)
        self.coordinates = self._extract_coordinates()

    def _extract_coordinates(self) -> dict[int, tuple[int, int]]:
        result = {}
        for i in self.problem.get_nodes():
            result[i] = self.problem.node_coords[i]
        return result
    
    def distance (self, a, b):
        x1, y1 = self.coordinates[a]
        x2, y2 = self.coordinates[b]
        if self.problem.edge_weight_type == "ATT":
            return att (x1, y1, x2, y2)
        if self.problem.edge_weight_type == "EUC_2D":
            return euc_2d (x1, y1, x2, y2)
    
    def optimal_tour (self, solution_path = "data/att48.opt.tour") -> list[int]:
        solution = tsplib95.load(solution_path)
        return solution.tours[0]

    def tour_length (self, tour: list[int]) -> int:
        total = 0
        """
        Tour i tour[1:] to kolejne elementy listy
        zip() robi krotki par tych elementów, np.:
        ex = ("a", "b", "c", "d")
        zip(ex, ex[1:]) == (('a', 'b'), ('b', 'c'), ('c', 'd'))
        """
        for u, v in zip (tour, tour[1:]):
            total += self.distance(u, v)
            
        if len(tour) > 1:
            total += self.distance(tour[-1], tour[0])

        return total