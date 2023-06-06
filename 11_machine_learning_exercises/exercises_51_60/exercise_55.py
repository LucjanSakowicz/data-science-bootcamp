# Dokonaj wektoryzacji dokumentów za pomocą klasy CountVectorizer z pakietu scikit-learn. Użyj argumentu stop_words i
# ustaw jego wartość na 'english'. Ustaw także odpowiedni argument, który pozwoli wydobyć n-gramy: unigramy i
# bigramy. Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)
df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)
print(df)
