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
