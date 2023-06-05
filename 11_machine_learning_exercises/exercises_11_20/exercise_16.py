# Przygotuj kolumnę investments do modelu, tzn. przekształć ją odpowiednio na typ int.
# W odpowiedzi wydrukuj obiekt DataFrame do konsoli.

import pandas as pd


df = pd.DataFrame(
    data={
        'investments': [
            '100_000_000',
            '100_000',
            '30_000_000',
            '100_500_000',
        ]
    }
)


df['investments'] = df['investments'].astype(int)
print(df)