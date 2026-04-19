Dodałem:

modularyzacje

możliwość wyboru algorytmu i funkcji celu

Czytanie z historii. Slider nie przyśpieszył tak bardzo jakbym tego chciał, ale jest jakaś poprawa

Nowe algorytmy tworzymy w sekcji "Algorytmy" i dodajemy tutaj 

eggholder

```python
# =========================
# WYBOR ALGORYTMU
# =========================
ALGORITHMS = {
    "pso": run_pso,
    "random": run_random_search,
}
```

Nowe funkcje celu tworzymy w sekcji funkcje celu i dodajemy w 
```python 
# =========================
# WYBÓR FUNKCJI CELU
# =========================
FUNCTIONS = {
    "rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
    },
    "eggholder": {
        "func": eggholder,
        "bounds": (-512, 512),
    }
}
```
Użytkownik używa tylko maina

1. Przygotowanie warstwy AG
2. AG dla funkcji ciągłych
3. AG dla TSP
4. Wpięcie AG do frameworka
5. Integracja z uruchamianiem i wynikami
6. Rozszerzenie wykresów pod AG
7. Warianty parametrów AG