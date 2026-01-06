import tsplib95
from typing import cast

# nearest integer
def nearest_int(result:float) -> int:
    return int(result + 0.5)

# algorytm att
def att (x1:int, y1:int, x2:int, y2:int) -> int:
    xd = x1 - x2
    yd = y1 - y2
    rij = ((xd*xd + yd*yd) / 10) ** 0.5
    tij = nearest_int (rij)
    dij = tij + 1 if tij < rij else tij
    return dij

# odległość euklidesowa
def euc_2d (x1: int, y1: int, x2: int, y2: int) -> int:
    xd = x1 - x2
    yd = y1 - y2
    dij = nearest_int ((xd*xd + yd*yd) ** 0.5)
    return dij

def optimal_tour (solution_path ="data/att48.opt.tour") -> list[int]:
    solution = tsplib95.load(solution_path)
    if solution.tours is None or len(solution.tours) == 0:
        raise ValueError("No solution found")
    tours = cast(list[list[int]], solution.tours)
    return tours[0]

class TSPProblem:
    def __init__(self, problem_path = "data/att48.tsp_ga") -> None:
        self.problem = tsplib95.load(problem_path)
        self.coordinates = self._extract_coordinates()

    def _extract_coordinates(self) -> dict[int, tuple[int, int]]:
        result: dict[int, tuple[int, int]] = {}
        coords = cast(dict[int, tuple[float, float]], self.problem.node_coords)

        for i in self.problem.get_nodes():
            x, y = coords[i]
            result[i] = (int(x), int(y))
        return result

    def distance (self, a, b) -> int:
        x1, y1 = self.coordinates[a]
        x2, y2 = self.coordinates[b]
        if self.problem.edge_weight_type == "ATT":
            return att (x1, y1, x2, y2)
        elif self.problem.edge_weight_type == "EUC_2D":
            return euc_2d (x1, y1, x2, y2)
        else:
            raise NotImplementedError (
                f"Nieobsługiwany typ odległości: {self.problem.edge_weight_type}"
            )

    def tour_length (self, tour: list[int]) -> int:
        total: int = 0

#        Tour i tour[1:] to kolejne elementy listy
#        zip() robi krotki par tych elementów, np.:
#        ex = ("a", "b", "c", "d")
#        zip(ex, ex[1:]) == (('a', 'b'), ('b', 'c'), ('c', 'd'))

        for u, v in zip (tour, tour[1:]):
            total = self.distance(u, v) + total
            
        if len(tour) > 1:
            total = self.distance(tour[-1], tour[0]) + total

        return total