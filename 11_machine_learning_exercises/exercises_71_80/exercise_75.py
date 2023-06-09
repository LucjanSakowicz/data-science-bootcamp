# Dokonaj kodowania 0-1 transakcji tak jak pokazano poniżej i przypisz do zmiennej transactions_encoded_df. Zmienną (
# obiekt DataFrame) transactions_encoded_df wydrukuj do konsoli.

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
print(transactions_encoded_df)