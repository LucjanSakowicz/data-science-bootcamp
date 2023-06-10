# Podane są poniższe obiekty: data_train target_train Dokonano pewnego przekształcenia zmiennych data_train oraz
# target_train. Wykorzystując klasę TfidfVectorizer z pakietu scikit-learn dokonaj wektoryzacji tekstu znajdującego
# się w liście data_train i przypisz do zmiennej data_train_vectorized. W odpowiedzi wydrukuj do konsoli kształt
# otrzymanej w ten sposób macierzy rzadkiej (sparse matrix).

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


data_train = pd.read_csv('data_train_91.csv')
target_train = pd.read_csv('target_train_91.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = TfidfVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)
print(data_train_vectorized.shape)