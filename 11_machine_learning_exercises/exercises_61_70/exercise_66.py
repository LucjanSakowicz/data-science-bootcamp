# Wykorzystując klasę DBSCAN z pakietu scikit-learn dokonaj podziału danych na klastry. Ustaw odpowiednio parametry:
# eps=0.6 min_samples=7 Dokonaj predykcji na podstawie tak zbudowanego modelu i przypisz numer klastra do każdej
# próbki w obiekcie df (nadaj nazwę kolumny cluster). Wyświetl rozkład częstości próbek w każdym klastrze.

import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN


df = pd.read_csv('clusters_66.csv')

model = DBSCAN(
    eps=0.6,
    min_samples=7
)
df['cluster'] = model.fit_predict(df)
print(df['cluster'].value_counts())
df = df['cluster'].value_counts().to_frame().reset_index()
df.columns = ['class', 'frequency']
px.bar(df, x='class', y='frequency', template='plotly_dark', title='Frequency per class',
       width=950, color_discrete_sequence=['#03fcb5']).show()
