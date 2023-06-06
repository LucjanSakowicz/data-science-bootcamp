# Wygenerowano zbiór raw_data zdefiniowany poniżej. Następnie podzielono go na zbiór treningowy i testowy.
# Wykorzystując klasę RandomForestClassifier z pakietu scikit-learn zbuduj model klasyfikacji dla podanych danych (
# ustaw argument random_state=42). Dokonaj trenowania modelu na zbiorze treningowym oraz oceny na zbiorze testowym. W
# odpowiedzi wydrukuj do konsoli dokładność modelu (do czterech miejsc po przecinku) na zbiorze testowym tak jak
# pokazano poniżej.


import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test):.4f}')
