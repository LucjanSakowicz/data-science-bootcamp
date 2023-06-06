# Zaimplementuj funkcję sigmoid() wykorzystując bibliotekę numpy. Korzystając z zaimplementowanej funkcji policz jej
# wartość dla zmiennej var1 i przypisz do kolumny o nazwie 'var1_sigmoid'. W odpowiedzi wyświetl obiekt DataFrame do
# konsoli.


import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame(data=np.random.randn(10), columns=['var1'])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


df['var1_sigmoid'] = df['var1'].apply(sigmoid)
print(df)
