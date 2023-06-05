# Poniżej podany jest obiekt DataFrame df: Wykorzystując klasę LabelEncoder z pakietu scikit-learn dokonaj kodowania
# 0-1 kolumny bought. Przypisz zmiany do obiektu df. W odpowiedzi wydrukuj obiekt df do konsoli.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes'],
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')

encoder = LabelEncoder()
df['bought'] = encoder.fit_transform(df['bought'])
print(df)