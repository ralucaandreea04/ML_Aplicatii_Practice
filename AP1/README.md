# Proiect: Predictia Soldului Energetic National

## Scopul
Acest proiect utilizeaza doua metode de predictie (ID3 si clasificare bayesiana) pentru estimarea soldului energetic in luna decembrie 2024 pe baza datelor istorice.

## Structura codului

1. **Importul datelor si prelucrarea**:
    - Datele sunt importate dintr-un fisier Excel.
    - Curatirea datelor elimina valorile nevalide si transforma coloanele in format numeric.

2. **Impartirea in seturi de antrenament si testare**:
    - Datele sunt impartite astfel incat decembrie sa fie utilizat exclusiv pentru testare.

3. **Antrenarea modelelor**:
    - Modelul ID3 utilizeaza un arbore de decizie pentru regresie.
    - Modelul Bayesian discretizeaza datele si aplica Bayes Naiv.

4. **Evaluarea performantei**:
    - Se calculeaza RMSE si MAE pentru ambele modele.
    - Se compara soldul real cu predictiile.
