import pandas as pd

# Lire le CSV directement
df = pd.read_csv("ventes_60j.csv", sep=",", encoding="utf-8")

# Sauvegarder en vrai fichier Excel
df.to_excel("ventes_60j.xlsx", index=False)

print("Conversion terminée : ventes_60j.xlsx créé")