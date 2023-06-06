# Wygenerowano zbiór raw_data zdefiniowany poniżej. Następnie podzielono go na zbiór treningowy (X_train, y_train) i
# testowy (X_test, y_test). Wykorzystując klasę DecisionTreeClassifier z pakietu scikit-learn zbuduj model
# klasyfikacji dla podanych danych. Wykorzystując metodę przeszukiwania siatki oraz klasę GridSearchCV (ustaw
# argumenty scoring='accuracy', cv=5) znajdź optymalne wartości parametrów max_depth oraz min_samples_leaf. Wartości
# parametrów przeszukaj z podanych poniżej: dla max_depth -> np.arange(1, 10) dla min_samples_leaf -> [1, 2, 3, 4, 5,
# 6, 7, 8, 9, 10, 15, 20] Dokonaj trenowania na zbiorze treningowym oraz oceny na zbiorze testowym. W odpowiedzi
# wydrukuj do konsoli najbardziej optymalne wartości parametrów max_depth oraz min_samples_leaf.


import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)
model = DecisionTreeClassifier()
parameters = {
    'max_depth': np.arange(1, 10),
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
}
clf = GridSearchCV(
    model, parameters, scoring='accuracy', cv=5
)
clf.fit(X_train, y_train)
print(clf.best_params_)
