# Oblicz wsparcie (support) dla pojedynczych produktów i wydrukuj do konsoli tak jak pokazano poniżej.
# Przypomnienie: Wsparcie(produkt A) = Liczba transakcji zawierających produkt A / łączna liczba transakcji w bazie

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

support = transactions_encoded_df.sum() / transactions_encoded_df.count()
print(support)