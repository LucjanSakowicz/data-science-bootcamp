# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera zmienne var_1, ..., var_10. Dokonaj analizy PCA
# wykorzystując pakiet scikit-learn. Zachowaj liczbę komponentów pozwalającą wyjaśnić 95% wariancji podanych danych.
# W odpowiedzi podaj liczbę komponentów uzyskanych w analizie.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
df = pd.read_csv('pca_72.csv')

scaler = StandardScaler()
X_std = scaler.fit_transform(df)

pca = PCA(n_components=0.95)
pca.fit(X_std)

print(f'Liczba komponentów: {pca.n_components_}')
