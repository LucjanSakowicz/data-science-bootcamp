# Wygenerowano zbiór raw_data zdefiniowany poniżej. Następnie podzielono go na zbiór treningowy i testowy.
# Wykorzystując klasę RandomForestClassifier z pakietu scikit-learn zbuduj model klasyfikacji dla podanych danych.
# Wykorzystując metodę przeszukiwania siatki oraz klasę GridSearchCV (ustaw argumenty scoring='accuracy',
# cv=5) znajdź optymalne wartości parametrów criterion, max_depth oraz min_samples_leaf. Wartości parametrów
# przeszukaj z podanych poniżej: dla criterion -> ['gini', 'entropy'] dla max_depth -> [6, 7, 8] dla min_samples_leaf
# -> [4, 5] Dokonaj trenowania na zbiorze treningowym oraz oceny na zbiorze testowym. W odpowiedzi wydrukuj do
# konsoli najbardziej optymalne wartości parametrów criterion,  max_depth oraz min_samples_leaf.

import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)
model = RandomForestClassifier(random_state=42)
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [6, 7, 8],
    'min_samples_leaf': [4, 5]
}
clf = GridSearchCV(
    model,
    param_grid=parameters,
    n_jobs=-1,
    scoring='accuracy',
    cv=2,
)
clf.fit(X_train, y_train)
print(clf.best_params_)
