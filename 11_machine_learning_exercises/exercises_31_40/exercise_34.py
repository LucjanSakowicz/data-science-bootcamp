# Podany jest poniższy obiekt DataFrame df: years  salary 0      1    4000 1      2    4250 2      3    4500 3      4
# 4750 4      5    5000 5      6    5250 Pierwsza kolumna opisuje lata pracy (zmienna objaśniająca), druga kolumna
# opisuje wynagrodzenie pracownika (zmienna objaśniana). Wykorzystując równanie normalne oraz pakiet numpy znajdź
# równanie regresji liniowej. Wynik wydrukuj do konsoli tak jak pokazano poniżej.

import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        'years': [1, 2, 3, 4, 5, 6],
        'salary': [4000, 4250, 4500, 4750, 5000, 5250],
    }
)
m = len(df)

X1 = df['years'].values
Y = df['salary'].values

X1 = X1.reshape(m, 1)
bias = np.ones((m, 1))
X = np.append(bias, X1, axis=1)

coefs = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
print(f'Linear regression: {coefs[0]:.2f} + {coefs[1]:.2f}x')