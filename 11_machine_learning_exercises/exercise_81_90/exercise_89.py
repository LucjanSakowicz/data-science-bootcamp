# Podane są dwa pliki: data_train.csv target_train.csv Plik data_train.csv zawiera maile dotyczące dwóch kategorii:
# grafiki komputerowej (comp.graphics) oraz przestrzeni kosmicznej (sci.space). Plik target_train.csv zawiera
# odpowiednio etykiety (0 - comp.graphics, 1 - sci.space). Wczytano zawartość plików jako obiekty DataFrame
# odpowiednio o nazwach: data_train target_train Dokonano pewnego przekształcenia zmiennych data_train oraz
# target_trian. Wykorzystując klasę CountVectorizer z pakietu scikit-learn dokonaj wektoryzacji tekstu znajdującego
# się w liście data_train i przypisz do zmiennej data_train_vectorized. W odpowiedzi wydrukuj do konsoli kształt
# otrzymanej w ten sposób macierzy rzadkiej (sparse matrix).

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


data_train = pd.read_csv('data_train_89.csv')
target_train = pd.read_csv('target_train_89.csv')

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = CountVectorizer()

data_train_vectorized = vectorizer.fit_transform(data_train)
print(data_train_vectorized.shape)