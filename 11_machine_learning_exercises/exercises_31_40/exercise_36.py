# Wczytaj dane z pliku data_36.csv do obiektu DataFrame. Następnie w oparciu o zmienną variable zbuduj model regresji
# liniowej pozwalający przewidywać wartości zmiennej docelowej target (model zbuduj na wszystkich dostępnych danych).
# Wykorzystaj w tym celu pakiet scikit-learn oraz klasę LinearRegression. Dokonaj oceny modelu wykorzystując score().
# Wynik wydrukuj do konsoli (do czwartego miejsca po przecinku).

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(pd.read_csv('data_36.csv'), columns=['variable', 'target'])
reg = LinearRegression()
reg.fit(df[['variable']], df[['target']])
print(round(reg.score(df[['variable']], df[['target']]), 4))
