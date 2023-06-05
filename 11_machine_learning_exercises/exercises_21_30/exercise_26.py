# Wykorzystując klasę OneHotEncoder z pakietu scikit-learn dokonaj kodowania 0-1 kolumny size (ustaw argument
# sparse=False). Wydrukuj zakodowaną postać kolumny size do konsoli (nie przypisuj zmian do zmiennej df). Wydrukuj
# także otrzymane kategorie przy kodowaniu kolumny size tak jak pokazano poniżej.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')
encoder = OneHotEncoder(sparse=False)
print(encoder.fit_transform(df[['size']]))
print(encoder.categories_)