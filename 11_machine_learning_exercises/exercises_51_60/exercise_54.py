# Dokonaj wektoryzacji dokumentów za pomocą klasy CountVectorizer z pakietu scikit-learn. Użyj argumentu stop_words i
# ustaw jego wartość na 'english'. Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python'
]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)
print(df)
