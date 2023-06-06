# Dokonaj wektoryzacji dokumentów wykorzystując klasę TfidfVectorizer z pakietu scikit-learn. Używając argumentu
# stop_words usuń z wektoryzacji dwa słowa: 'is' oraz 'in'. Wynik wyświetl w postaci obiektu DataFrame tak jak
# pokazano poniżej.

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
pd.set_option('precision', 3)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]

vectorizer = TfidfVectorizer(stop_words=['is', 'in'])
X = vectorizer.fit_transform(documents)
df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)
print(df)