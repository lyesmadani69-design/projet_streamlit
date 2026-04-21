from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:

    HOLT_OK = True
except Exception:
    HOLT_OK = False


# =========================================================
# CONFIGURATION DE L'APP
# =========================================================
st.set_page_config(
    page_title="Lyes Data Freelance",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #bfc7d5;
        margin-bottom: 1rem;
    }
    .section-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(0, 200, 120, 0.20);
        background: rgba(0, 200, 120, 0.06);
        margin-bottom: 1rem;
    }
    .insight-box {
        padding: 0.85rem 1rem;
        border-left: 4px solid #00c878;
        background: rgba(0, 200, 120, 0.08);
        border-radius: 10px;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CHEMINS & DOSSIERS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

for folder in [ASSETS_DIR, DATA_DIR, UPLOADS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS SIMPLES
# =========================================================
def euro(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.2f} €".replace(",", " ").replace(".", ",")


def pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1f} %"


def safe_div(a, b):
    return a / b if b not in (0, None) else 0


def resolve_asset(filename: str) -> Path | None:
    if not filename:
        return None
    path = ASSETS_DIR / filename
    return path if path.exists() else None


# =========================================================
# PIPELINE - PREPARATION DONNEES
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace("é", "e")
        .str.replace("è", "e")
        .str.replace("ê", "e")
        .str.replace("à", "a")
        .str.replace(" ", "_")
    )
    return df


def clean_cash_data(df: pd.DataFrame):
    df = normalize_columns(df)

    stats = {
        "lignes_initiales": len(df),
        "doublons": 0,
        "valeurs_manquantes": 0,
    }

    if "prix_unitaire_ttc" in df.columns:
        df["prix_unitaire_ttc"] = (
            df["prix_unitaire_ttc"]
            .astype(str)
            .str.replace("€", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df["prix_unitaire_ttc"] = pd.to_numeric(
            df["prix_unitaire_ttc"], errors="coerce"
        )

    if "quantite" in df.columns:
        df["quantite"] = pd.to_numeric(df["quantite"], errors="coerce")
        df["quantite"] = df["quantite"].fillna(1)

    if "date_vente" in df.columns:
        df["date_vente"] = pd.to_datetime(
            df["date_vente"], errors="coerce", dayfirst=True
        )

    if "produit" in df.columns:
        df["produit"] = df["produit"].astype(str).str.strip()

    if "categorie" in df.columns:
        df["categorie"] = df["categorie"].replace(
            {
                "Boisson froide": "Boissons froides",
            }
        )

    if "heure" in df.columns:
        df["heure_num"] = df["heure"].astype(str).str[:2]
        df["heure_num"] = pd.to_numeric(df["heure_num"], errors="coerce")

    before = len(df)
    required_cols = [c for c in ["prix_unitaire_ttc", "quantite"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)
    stats["valeurs_manquantes"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    stats["doublons"] = before - len(df)

    if {"prix_unitaire_ttc", "quantite"}.issubset(df.columns):
        df["ca_ligne"] = df["prix_unitaire_ttc"] * df["quantite"]

    stats["lignes_finales"] = len(df)

    return df, stats


# =========================================================
# SIDEBAR - POSITIONNEMENT PRO
# =========================================================
with st.sidebar:
    st.markdown("## Lyes Data Freelance")

    st.write(
        "J’analyse vos données de vente pour identifier ce qui vous fait gagner ou perdre de l’argent."
    )

    st.markdown("---")
    st.write("📍 Basé à Lyon")
    st.write("📊 Spécialiste : commerces de proximité")
    st.write("📦 Données : caisse / ventes / produits")

    st.markdown("---")
    st.markdown("### Ce que je vous apporte")
    st.write("✔ Identifier vos produits rentables")
    st.write("✔ Détecter les pertes et anomalies")
    st.write("✔ Améliorer votre panier moyen")
    st.write("✔ Prendre de meilleures décisions")

    st.markdown("---")
    st.markdown("### Me contacter")

    st.link_button(
        "📲 WhatsApp",
        "https://wa.me/33763766454",
        use_container_width=True,
    )

    st.link_button(
        "💼 LinkedIn",
        "https://www.linkedin.com/in/TON_PROFIL",
        use_container_width=True,
    )

    st.link_button(
        "🌐 GitHub",
        "https://github.com/lyesmadani69-design",
        use_container_width=True,
    )

    cv_path = ASSETS_DIR / "CV.pdf"
    if cv_path.exists():
        with open(cv_path, "rb") as f:
            st.download_button(
                "📄 Télécharger mon CV",
                f,
                file_name="CV_Lyes_Data.pdf",
                use_container_width=True,
            )

    st.markdown("---")
    st.caption("Transformez vos données en chiffre d’affaires 📈")


# =========================================================
# STRUCTURE PRINCIPALE
# =========================================================
st.markdown(
    """
    <div class="section-box">
        <div class="main-title">Transformez vos données en décisions rentables 📊</div>
        <div class="subtitle">
            Peu importe votre activité : si vos données existent, même mal organisées,
            je peux les structurer, les analyser et en tirer des actions concrètes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_pipeline, tab_demo, tab_projects, tab_contact = st.tabs(
    [
        "Accueil",
        "Pipeline",
        "Démo CSV",
        "Projets",
        "Contact",
    ]
)

# =========================================================
# ONGLET DEMO CSV
# =========================================================
with tab_demo:
    st.subheader("Démo : transformation d’un export de caisse")

    st.markdown(
        """
        Importez un fichier CSV brut ou utilisez le fichier de démonstration
        pour voir comment il peut être transformé en données propres, lisibles
        et directement exploitables.
        """
    )

    demo_path = DATA_DIR / "demo.csv"

    use_demo = st.checkbox("Utiliser le fichier de démonstration", value=True)

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        key="demo_csv_uploader",
    )

    df_raw = None

    if use_demo and demo_path.exists():
        df_raw = pd.read_csv(demo_path)
        st.caption(f"Fichier de démonstration chargé : {demo_path.name}")

    elif uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            df_raw = pd.read_csv(uploaded_file)

    if df_raw is not None:
        st.markdown("### Aperçu brut")
        st.dataframe(df_raw.head(10), use_container_width=True)

        df_clean, stats = clean_cash_data(df_raw)

        st.markdown("### Résultat du nettoyage")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes initiales", stats["lignes_initiales"])
        c2.metric("Lignes finales", stats["lignes_finales"])
        c3.metric("Doublons supprimés", stats["doublons"])
        c4.metric("Valeurs manquantes traitées", stats["valeurs_manquantes"])

        st.markdown("### Aperçu nettoyé")
        st.dataframe(df_clean.head(20), use_container_width=True)

        if "ca_ligne" in df_clean.columns:
            st.markdown("### Indicateurs clés")
            k1, k2, k3 = st.columns(3)

            k1.metric("Chiffre d’affaires", euro(df_clean["ca_ligne"].sum()))
            k2.metric("CA moyen par ligne", euro(df_clean["ca_ligne"].mean()))

            if "produit" in df_clean.columns:
                k3.metric("Produits distincts", int(df_clean["produit"].nunique()))
            else:
                k3.metric("Produits distincts", "N/A")

        if {"produit", "ca_ligne"}.issubset(df_clean.columns):
            st.markdown("### Top produits")
            top_products = (
                df_clean.groupby("produit", as_index=False)["ca_ligne"]
                .sum()
                .sort_values("ca_ligne", ascending=False)
                .head(10)
            )
            st.dataframe(top_products, use_container_width=True)

        if {"categorie", "ca_ligne"}.issubset(df_clean.columns):
            st.markdown("### Répartition par catégorie")
            top_categories = (
                df_clean.groupby("categorie", as_index=False)["ca_ligne"]
                .sum()
                .sort_values("ca_ligne", ascending=False)
            )
            st.dataframe(top_categories, use_container_width=True)

        if "moyen_paiement" in df_clean.columns:
            st.markdown("### Répartition des paiements")
            paiements = df_clean["moyen_paiement"].value_counts().reset_index()
            paiements.columns = ["moyen_paiement", "nombre"]
            st.dataframe(paiements, use_container_width=True)

        if {"heure_num", "ca_ligne"}.issubset(df_clean.columns):
            st.markdown("### Activité par heure")
            hourly = (
                df_clean.groupby("heure_num", as_index=False)["ca_ligne"]
                .sum()
                .sort_values("heure_num")
            )
            st.line_chart(hourly.set_index("heure_num"))

        st.markdown("### Lecture simple")
        st.success(
            "À partir d’un fichier brut, je peux reconstituer une base propre, "
            "calculer le chiffre d’affaires, faire ressortir les produits clés, "
            "les catégories dominantes et préparer une analyse commerciale fiable."
        )

        csv_clean = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le CSV nettoyé",
            data=csv_clean,
            file_name="sortie_caisse_nettoyee.csv",
            mime="text/csv",
        )

    else:
        st.info("Importez un fichier CSV ou activez le fichier de démonstration.")

# =========================================================
# ONGLET PROJETS
# =========================================================
with tab_projects:
    st.subheader("Exemples de projets")

    st.markdown(
        """
        Voici des cas d’usage typiques où mes analyses permettent
        de transformer des données brutes en décisions concrètes.
        """
    )

    st.markdown("### 1. Commerce de proximité / bar-tabac")
    st.markdown(
        """
        **Objectif :** comprendre ce qui génère réellement du chiffre d’affaires
        et ce qui pèse sur la rentabilité.

        **Ce que l’analyse permet de faire :**
        - identifier les produits les plus rentables
        - comparer les catégories
        - repérer les heures et jours les plus performants
        - mettre en évidence les anomalies ou incohérences
        - proposer des actions pour augmenter le panier moyen
        """
    )

    st.markdown("### 2. Restaurant / snack")
    st.markdown(
        """
        **Objectif :** analyser les ventes pour optimiser les menus,
        les produits complémentaires et les heures de forte activité.

        **Ce que l’analyse permet de faire :**
        - identifier les articles les plus vendus
        - distinguer volume et rentabilité
        - détecter les produits peu performants
        - proposer des leviers simples d’upsell
        """
    )

    st.markdown("### 3. Salon de beauté / commerce de services")
    st.markdown(
        """
        **Objectif :** mieux comprendre quelles prestations et quels produits
        contribuent le plus au chiffre d’affaires.

        **Ce que l’analyse permet de faire :**
        - mesurer la rentabilité par prestation
        - comparer les ventes de produits associés
        - détecter les opportunités de ventes complémentaires
        - structurer une lecture claire de l’activité
        """
    )

    st.markdown("### Ce que j’apporte sur chaque projet")
    st.markdown(
        """
        - une base de données nettoyée et fiable  
        - des indicateurs compréhensibles  
        - des visualisations utiles  
        - des recommandations directement exploitables  
        """
    )

#  =========================================================
# ONGLET ACCUEIL
# =========================================================
with tab_home:
    st.subheader("Bienvenue")

    st.markdown(
        """
        J’aide les commerces à mieux exploiter leurs données de vente pour répondre à des questions concrètes :

        - Quels produits rapportent vraiment ?
        - Quels articles sont peu rentables ?
        - À quelles heures ou quels jours l’activité est la plus forte ?
        - Comment augmenter le panier moyen sans attirer plus de clients ?
        """
    )

    st.markdown("### Ce que j’apporte")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="insight-box">
            ✔️ Nettoyage et structuration de données brutes<br>
            ✔️ Contrôle qualité avant analyse<br>
            ✔️ Mise en évidence des produits et catégories rentables
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="insight-box">
            ✔️ Analyse des ventes et du panier moyen<br>
            ✔️ Détection d’anomalies et d’opportunités<br>
            ✔️ Restitution claire avec recommandations
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="insight-box">
            ✔️ Analyse des ventes et du panier moyen<br>
            ✔️ Détection d’anomalies et d’opportunités<br>
            ✔️ Restitution claire avec recommandations
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 🔒 BLOC CONFIDENTIALITÉ (À AJOUTER ICI)
    st.markdown("### Confidentialité & protection des données")

    st.markdown(
        """
        <div class="section-box">
            <strong>🔒 Vos données sont traitées avec sérieux et confidentialité</strong><br><br>
            En tant que freelance data analyst, je traite les fichiers confiés dans un cadre
            professionnel, avec une attention particulière à la sécurité et à la discrétion.<br><br>

            ✔️ Utilisation des données uniquement dans le cadre de la mission<br>
            ✔️ Aucun partage ni revente à des tiers<br>
            ✔️ Manipulation sécurisée des fichiers transmis<br>
            ✔️ Suppression possible des données sur demande<br>
            ✔️ Accord de confidentialité (NDA) possible avant toute analyse<br><br>

            Mon approche s’appuie sur les principes du RGPD ainsi que sur les bonnes pratiques
            recommandées par la CNIL, afin de vous permettre de confier vos données en toute confiance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Exemple concret")
    st.markdown(
        """
        Dans un commerce, une hausse de seulement **+1 € par ticket**
        peut représenter plusieurs milliers d’euros supplémentaires sur l’année.

        L’objectif n’est pas de produire des graphiques pour faire joli,
        mais d’aider à prendre de meilleures décisions.
        """
    )

    st.markdown("### Pour quels commerces ?")
    st.write(
        "Bar, tabac, snack, restaurant, salon de beauté, commerce de proximité… "
        "Dès lors que les données existent, elles peuvent être structurées et exploitées."
    )


# =========================================================
# ONGLET PIPELINE (METHODOLOGIE)
# =========================================================
with tab_pipeline:
    st.subheader("Méthodologie d’analyse")

    st.markdown(
        """
        J’utilise une méthode structurée pour transformer un simple export de caisse
        en décisions concrètes et exploitables.
        """
    )

    st.markdown("### 1. Chargement des données")
    st.markdown(
        """
        Je récupère l’export de caisse et je vérifie sa structure :
        colonnes disponibles, volume de données, dates, produits et tickets.
        """
    )

    st.markdown("### 2. Contrôle qualité (EDA informative)")
    st.markdown(
        """
        Avant toute analyse, je contrôle la qualité des données :

        - doublons
        - valeurs manquantes
        - incohérences de format
        - vérification de la granularité (ticket / ligne produit)
        """
    )

    st.markdown("### 3. Nettoyage des données")
    st.markdown(
        """
        Je rends les données fiables :

        - correction des formats (prix, dates, quantités)
        - suppression des anomalies évidentes
        - uniformisation des colonnes
        """
    )

    st.markdown("### 4. Analyse exploratoire")
    st.markdown(
        """
        Je fais parler les données pour faire ressortir les tendances :

        - produits les plus vendus
        - catégories dominantes
        - panier moyen
        - heures et jours les plus performants
        """
    )

    st.markdown("### 5. Règles métier (engine rules)")
    st.markdown(
        """
        J’adapte l’analyse à la réalité du commerce :

        - certains produits ont une marge faible (ex : tabac)
        - d’autres fonctionnent à la commission (FDJ, services)
        - vérification de la cohérence des prix et des catégories
        """
    )

    st.markdown("### 6. Insights & visualisation")
    st.markdown(
        """
        Je transforme les données en informations utiles :

        - produits rentables à mettre en avant
        - produits à faible marge à surveiller
        - opportunités d’augmentation du panier moyen
        """
    )

    st.markdown("### 7. Restitution")
    st.markdown(
        """
        Je fournis une synthèse claire avec des recommandations concrètes,
        directement exploitables pour améliorer le chiffre d’affaires.
        """
    )
# =========================================================
# ONGLET CONTACT
# =========================================================
with tab_contact:
    st.subheader("Contact")

    st.markdown(
        """
        Vous avez un export de caisse, un fichier de ventes ou des données peu exploitables ?

        Je peux vous aider à les structurer, les analyser et en tirer des actions concrètes.
        """
    )

    st.markdown("### Me joindre")
    st.write("📧 Email : lyesmadani69@gmail.com")
    st.write("📍 Basé à Lyon")
    st.write("💬 Analyse gratuite possible selon le besoin")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.link_button(
            "📲 WhatsApp",
            "https://wa.me/33763766454",
            use_container_width=True,
        )

    with c2:
        st.link_button(
            "💼 LinkedIn",
            "https://www.linkedin.com/in/TON_PROFIL",
            use_container_width=True,
        )

    with c3:
        st.link_button(
            "🌐 GitHub",
            "https://github.com/lyesmadani69-design",
            use_container_width=True,
        )

    st.markdown("---")
    st.success(
        "Peu importe votre activité : si les données existent, elles peuvent être structurées et exploitées."
    )
