# Proiect: Predictia Soldului Energetic National

## Scopul
Acest proiect utilizeaza doua metode de predictie (ID3 si clasificare bayesiana) pentru estimarea soldului energetic in luna decembrie 2024 pe baza datelor istorice.

## Structura codului

1. **Importul datelor si prelucrarea**:
    - Datele sunt importate dintr-un fisier Excel. (liniile 10,11)
    - Curatirea datelor elimina valorile nevalide si transforma coloanele in format numeric.(liniile 16-22)

2. **Impartirea in seturi de antrenament si testare**:
    - Datele sunt impartite astfel incat decembrie sa fie utilizat exclusiv pentru testare. (liniile 24-45)

3. **Antrenarea modelelor**:
    - Modelul ID3 utilizeaza un arbore de decizie pentru regresie. (liniile 47-50)
    - Modelul Bayesian discretizeaza datele si aplica Bayes Naiv. (liniile 57-64)

4. **Evaluarea performantei**:
    - Se calculeaza RMSE si MAE pentru ambele modele. (liniile 52-55 si 66-69)
    - Se compara soldul real cu predictiile. (liniile 71-90)
