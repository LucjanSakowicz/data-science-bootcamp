# Oblicz wsparcie dla par: (butter, bread) oraz (butter, milk). Wynik wydrukuj do konsoli (zaokrąglij wsparcie do
# czterech miejsc po przecinku). Przypomnienie: Wsparcie(produkt A, produkt B) = Liczba transakcji zawierających
# produkt A i produkt B / łączna liczba transakcji w bazie


import numpy as np
import pandas as pd

data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

transactions_encoded = np.zeros(
    (len(transactions), len(products)), dtype='int8'
)

for row in zip(
        range(len(transactions)), transactions_encoded, expanded.values
):
    for idx, product in enumerate(products):
        if product in row[2]:
            transactions_encoded[row[0], idx] = 1
transactions_encoded_df = pd.DataFrame(
    transactions_encoded, columns=products
)

support_butter_bread = len(transactions_encoded_df[(transactions_encoded_df['butter'] == 1) & (
            transactions_encoded_df['bread'] == 1)]) / len(transactions_encoded_df)
support_butter_milk = len(transactions_encoded_df[(transactions_encoded_df['butter'] == 1) & (
            transactions_encoded_df['milk'] == 1)]) / len(transactions_encoded_df)
print(f'support(butter, bread) = {support_butter_bread:.4f}')
print(f'support(butter, milk) = {support_butter_milk:.4f}')

