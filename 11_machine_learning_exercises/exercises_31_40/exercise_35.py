# Podany jest poniższy obiekt DataFrame df: years  salary 0      1    4000 1      2    4250 2      3    4500 3      4
# 4750 4      5    5000 5      6    5250 Pierwsza kolumna opisuje lata pracy (zmienna objaśniająca), druga kolumna
# opisuje wynagrodzenie pracownika (zmienna objaśniana). Wykorzystując pakiet scikit-learn oraz klasę
# LinearRegression znajdź równanie regresji liniowej dla tego problemu. Wynik wydrukuj do konsoli tak jak pokazano
# poniżej.

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(
    {
        'years': [1, 2, 3, 4, 5, 6],
        'salary': [4000, 4250, 4500, 4750, 5000, 5250],
    }
)

reg = LinearRegression()
reg.fit(df[['years']], df[['salary']])
print(f'Linear regression: {reg.intercept_[0]:.2f} + {reg.coef_.ravel()[0]:.2f}x')