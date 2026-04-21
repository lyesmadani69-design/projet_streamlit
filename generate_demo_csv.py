from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "demo.csv"

products = [
    {"produit": "Café expresso", "categorie": "Boissons chaudes", "prix": 1.30},
    {"produit": "Café allongé", "categorie": "Boissons chaudes", "prix": 1.50},
    {"produit": "Thé menthe", "categorie": "Boissons chaudes", "prix": 2.00},
    {"produit": "Coca-Cola 33cl", "categorie": "Boissons froides", "prix": 2.50},
    {"produit": "Eau 50cl", "categorie": "Boissons froides", "prix": 1.00},
    {"produit": "Red Bull", "categorie": "Boissons froides", "prix": 2.80},
    {"produit": "Croissant", "categorie": "Snacking", "prix": 1.20},
    {"produit": "Pain au chocolat", "categorie": "Snacking", "prix": 1.30},
    {"produit": "Sandwich jambon beurre", "categorie": "Snacking", "prix": 4.80},
    {"produit": "Panini fromage", "categorie": "Snacking", "prix": 5.50},
    {"produit": "Marlboro Red", "categorie": "Tabac", "prix": 12.50},
    {"produit": "Camel Filters", "categorie": "Tabac", "prix": 11.90},
    {"produit": "Lucky Strike", "categorie": "Tabac", "prix": 12.20},
    {"produit": "Grattage Astro", "categorie": "FDJ", "prix": 5.00},
    {"produit": "Euromillions", "categorie": "FDJ", "prix": 2.50},
    {"produit": "Briquet Bic", "categorie": "Accessoires", "prix": 1.50},
    {"produit": "Feuilles OCB", "categorie": "Accessoires", "prix": 1.20},
]

rows = []
ticket_counter = 1000

dates = pd.date_range("2024-01-01", periods=60, freq="D")

for current_date in dates:
    weekday = current_date.weekday()
    tickets_per_day = random.randint(35, 65)
    if weekday in [4, 5]:
        tickets_per_day += random.randint(10, 20)

    for _ in range(tickets_per_day):
        ticket_counter += 1
        ticket_id = f"T{ticket_counter}"

        nb_lines = random.choices([1, 2, 3, 4], weights=[25, 40, 25, 10], k=1)[0]

        hour = random.choices(
            population=list(range(6, 23)),
            weights=[2, 3, 5, 6, 7, 7, 6, 5, 4, 4, 5, 6, 8, 9, 9, 8, 6],
            k=1,
        )[0]
        minute = random.randint(0, 59)
        heure = f"{hour:02d}:{minute:02d}"

        moyen_paiement = random.choices(["CB", "Espèces"], weights=[65, 35], k=1)[0]
        selected_products = random.sample(products, k=nb_lines)

        for item in selected_products:
            quantite = random.choices([1, 2, 3], weights=[75, 20, 5], k=1)[0]

            rows.append(
                {
                    "date_vente": current_date.strftime("%Y-%m-%d"),
                    "heure": heure,
                    "ticket_id": ticket_id,
                    "produit": item["produit"],
                    "categorie": item["categorie"],
                    "quantite": quantite,
                    "prix_unitaire_ttc": item["prix"],
                    "moyen_paiement": moyen_paiement,
                }
            )

df = pd.DataFrame(rows)

# Doublons légers
duplicates = df.sample(12, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)


# Prix texte avec virgule
df["prix_unitaire_ttc"] = df["prix_unitaire_ttc"].astype("object")
idx_price_text = df.sample(20, random_state=1).index
df.loc[idx_price_text, "prix_unitaire_ttc"] = (
    df.loc[idx_price_text, "prix_unitaire_ttc"]
    .astype(str)
    .str.replace(".", ",", regex=False)
)

# Quantités manquantes
idx_qty_missing = df.sample(15, random_state=2).index
df.loc[idx_qty_missing, "quantite"] = None

# Variantes catégories
cold_idx = df[df["categorie"] == "Boissons froides"].sample(12, random_state=3).index
df.loc[cold_idx, "categorie"] = "Boisson froide"

# Variantes produits
coffee_idx = df[df["produit"] == "Café expresso"].sample(10, random_state=4).index
df.loc[coffee_idx, "produit"] = "Cafe expresso"

# Heures manquantes
hour_idx = df.sample(8, random_state=5).index
df.loc[hour_idx, "heure"] = None

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Fichier créé : {OUTPUT_PATH}")
print(f"Nombre de lignes : {len(df)}")
