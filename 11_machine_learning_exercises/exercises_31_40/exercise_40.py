# Plik predictions.csv zawiera predykcje pewnego modelu regresji: zmienna y_true opisuje rzeczywiste, zaobserwowane
# wartości zmienna y_pred opisuje wartości przewidziane przez model Wczytaj ten plik do obiektu DataFrame. Następnie
# zaimplementuj funkcję o nazwie mean_squared_error() obliczającą błąd średniokwadratowy predykcji. Wykorzystując
# zaimplementowaną funkcję policz wartość MSE dla tego modelu. Wynik wydrukuj do konsoli tak jak pokazano poniżej.

import numpy as np
import pandas as pd


def mean_squared_error(y_true, y_pred):
    return np.power(y_true - y_pred, 2).sum() / len(y_true)


df = pd.DataFrame(pd.read_csv('predictions.csv'))
mae = mean_squared_error(df['y_true'], df['y_pred'])
print(f'MSE = {mae:.4f}')