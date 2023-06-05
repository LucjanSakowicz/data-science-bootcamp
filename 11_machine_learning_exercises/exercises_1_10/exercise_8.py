# Dokonaj dyskretyzacji kolumny weight na 3 przedziały o równej szerokości. Wynik przypisz do nowej kolumny o nazwie
# 'weight_cut' tak jak pokazano poniżej. W odpowiedzi wydrukuj obiekt df do konsoli.


import pandas as pd


df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})
df['weight_cut'] = pd.cut(df['weight'], bins=3)
print(df)