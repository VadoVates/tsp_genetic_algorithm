"""
genetic_algorithm.py – pętla GA: inicjalizacja populacji,
selekcja, krzyżowanie, mutacja, elityzm,
ocena fitness (używa tsp_problem.tour_length).
Zwraca najlepszą trasę i historię postępu.
"""

def euc_2d(result:float):
    try:
        if result > 0:
            integer_result = int (result)
            if result - integer_result < 0.5:
                return integer_result
            else:
                return integer_result+1
        else:
            print ("ODLEGŁOŚĆ <= 0, nieobsługiwany wyjątek")
            return -1
    except Exception as e:
        print (f"Nieprawidłowe dane wejściowe, treść błędu: {e}")
        return -1
    
def att(x1:int, y1:int, x2:int, y2:int):
    xd = x1 - x2
    yd = y1 - y2
    r = ((xd*xd + yd*yd)/10.0) ** 0.5   # zgodnie z algorytmem problemu tsp dodane dzielenie przez 10 celem skalowania dużych odległości
    t = euc_2d(r)
    return t

print (att(0,0,200,200))