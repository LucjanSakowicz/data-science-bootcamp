# Wytnij wszystkie wiersze obiektu df dla których kolumna weight nie przyjmuje wartość np.nan. Na tak otrzymanym
# obiekcie policz wartość średnią dla kolumn numerycznych price oraz weight. Wynik wydrukuj do konsoli.

import numpy as np
import pandas as pd


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)
print(df[~df['weight'].isnull()][['price', 'weight']].mean())