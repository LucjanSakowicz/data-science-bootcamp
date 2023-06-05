# Podane są poniższe tablice: data target Połączono te dwie tablice w jedną o nazwie all_data. Utwórz z tablicy
# all_data obiekt DataFrame nadając odpowiednio nazwy kolumn (zawartość klucza 'feature_names' obiektu raw_data +
# nazwa zmiennej docelowej jako 'target'). W odpowiedzi wydrukuj obiekt DataFrame do konsoli.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']
all_data = np.c_[data, target]
df = pd.DataFrame(
    all_data,
    columns=np.append(raw_data['feature_names'], 'target')
)
print(df.head(5))