# Dokonaj kodowania 0-1 obiektu df (dokładnie kolumny weight_cut) dzięki funkcji pd.get_dummies(). W odpowiedzi wynik
# kodowania wydrukuj do konsoli.

import pandas as pd


df = pd.DataFrame(
    data={'weight': [75.0, 78.5, 85.0, 91.0, 84.5, 83.0, 68.0]}
)
df['weight_cut'] = pd.cut(
    df['weight'],
    bins=(60, 75, 80, 95),
    labels=['light', 'normal', 'heavy'],
)
print(pd.get_dummies(df))