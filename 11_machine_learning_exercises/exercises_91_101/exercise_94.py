# Wyświetl korelację zmiennych ze zmienną docelową target (w kolejności malejącej). Wynik wydrukuj do konsoli tak jak
# pokazano poniżej.

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
print(df.corr()['target'].sort_values(ascending=False)[1:])