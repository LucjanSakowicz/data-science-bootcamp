# Zaimplementuj funkcję entropy() wykorzystując pakiet numpy. Wynik zaokrąglij do czwartego miejsca po przecinku.
# Możesz oprzeć się o poniższy wzór: Następnie podany jest poniższy obiekt DataFrame: val_1  val_2 0   0.01   0.99 1
# 0.11   0.89 2   0.21   0.79 3   0.31   0.69 4   0.41   0.59 5   0.51   0.49 6   0.61   0.39 7   0.71   0.29 8
# 0.81   0.19 9   0.91   0.09 Wykorzystując zaimplementowaną funkcję entropy() wyznacz trzecią kolumnę obiektu
# DataFrame o nazwie 'entropy' zawierającą entropię dla poszczególnych wierszy. Tak utworzony obiekt wydrukuj do
# konsoli.

import numpy as np
import pandas as pd


def entropy(x):
    return np.round(-np.sum(x * np.log2(x)), 4)


df = pd.DataFrame(
    {
        'val_1': np.arange(0.01, 1.0, 0.1),
        'val_2': 1 - np.arange(0.01, 1.0, 0.1),
    }
)


def entropy(x):
    return np.round(-np.sum(x * np.log2(x)), 4)


df['entropy'] = [
    entropy([row[1][0], row[1][1]]) for row in df.iterrows()
]
print(df)
