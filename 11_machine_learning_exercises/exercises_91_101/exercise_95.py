# Skopiuj obiekt df do zmiennej data. Następnie wyrwij kolumnę target ze zmiennej data i przypisz do zmiennej target.
# Wyświetl pięć pierwszych wierszy obiektu data, następnie wydrukuj pustą linię i kolejno pięć pierwszych wierszy
# obiektu target tak jak pokazano poniżej.

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)
target = df.pop('target')
print(df.head(5))
print()
print(target.head(5))
