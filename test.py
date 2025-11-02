import os
import random
from typing import List, Tuple

""" KODOWANIE / DEKODOWANIE """

def pack (x1: int, x2: int) -> int:
    #x1 i x2 z zakresu [0..31] połączone razem
    try:
        if x1 > 31 or x2 > 31 or x1 < 0 or x2 < 0:
            print (f"Wewnątrz funkcji pack, zła wartość którejś zmiennej. x1: {x1}, x2: {x2}.")
    except Exception as e:
        print (f"Wewnątrz funkcji pack, nieprawidłowe dane wejściowe. {e}")
    return (x1 << 5 | x2)

def unpack (chromosome: int) -> Tuple [int, int]:
    try:
        if chromosome >= 1024 or chromosome < 0:
            print (f"Wewnątrz funkcji unpack, zbyt duża wartość wejściowa chromosome: {chromosome}. Kod zostanie wykonany dalej.") 
        x2 = chromosome & 0b11111
        x1 = (chromosome >> 5) & 0b11111
    except Exception as e:
        print (f"Wewnątrz funkcji unpack, nieprawidłowe dane wejściowe. {e}")
    return x1, x2

""" FITNESS / FUNKCJA CELU """

def f (chromosome: int) -> int:
    x1, x2 = unpack (chromosome)
    return x1 - x2

def g (chromosome: int) -> int:
    return f (chromosome) + 32

""" ROULETTE SELECTION """

def roulette_select (population: list[int], count: int, return_counts = False):

    fit_list = [g(y) for y in population]
    total = sum (fit_list)

    if total == 0: # jeżeli wszystkie dają różnicę zero, wtedy wybierz losowo
        return random.choices (population, k = count)

    fits_accumulated = []
    s = 0
    for fit in fit_list:
        s+=fit
        fits_accumulated.append(s)
    
    # print (f"Szanse na wylosowanie to:")
    # for element in fit_list:
    #     print (element/total)

    selected = []
    for _ in range (count):
        r = random.randint (0, total)
        for i, accumulation in enumerate (fits_accumulated):
            if r <= accumulation:
                selected.append(population[i])
                break

    return selected

if __name__ == "__main__":
    population = [
        0b1000110000,
        0b1010110101,
        0b0010100101,
        0b0000100001,
        0b0001100011
        ]
    population.sort()
    print (population)
    print(roulette_select (population, 4))
    # print ("{:10b}".format(pack(27,22))) # drukuj binarny wynik: 10 znaków
    # run_ga(generations = 100, p_cross = 0.5, p_mut = 0.02, trace_first = 1)