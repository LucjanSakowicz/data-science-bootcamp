# Podane są poniższe obiekty: data_train target_train Dokonano pewnego przekształcenia zmiennych data_train oraz
# target_train. Wykorzystano klasę CountVectorizer z pakietu scikit-learn do wektoryzacji tekstu i przypisano do
# zmiennej data_train_vectorized. Wykorzystując klasę MultinomialNB zbuduj model klasyfikacji dokumentów tekstowych.
# Model wytrenuj w oparciu o dane data_train_vectorized oraz target_train. Następnie dokonaj klasyfikacji poniższych
# zdań: 'The graphic designer requires a good processor to work.' 'Flights into space.' Wynik wydrukuj do konsoli tak
# jak pokazano poniżej.

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data_train = pd.read_csv('data_train_90.csv')
target_train = pd.read_csv('target_train_90.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = CountVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)

classifier = MultinomialNB()
classifier.fit(data_train_vectorized, target_train)

docs = [
    'The graphic designer requires a good processor to work',
    'Flights into space',
]
data_new = vectorizer.transform(docs)

data_pred = classifier.predict(data_new)

for doc, category in zip(docs, data_pred):
    print(f'\'{doc}\' => {categories[category]}')
