from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False


# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="Lyes Data - Dashboard commercial",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #cfcfcf;
        margin-bottom: 1rem;
    }
    .hero-box {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 200, 120, 0.25);
        background: linear-gradient(135deg, rgba(0,200,120,0.08), rgba(255,255,255,0.02));
        margin-bottom: 1rem;
    }
    .insight-box {
        padding: 0.9rem 1rem;
        border-left: 4px solid #00c878;
        background: rgba(0,200,120,0.08);
        border-radius: 10px;
        margin-bottom: 0.7rem;
    }
    .risk-box {
        padding: 0.9rem 1rem;
        border-left: 4px solid #ff6b6b;
        background: rgba(255,107,107,0.08);
        border-radius: 10px;
        margin-bottom: 0.7rem;
    }
    .cta-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(0,200,120,0.22);
        background: rgba(0,200,120,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
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


def compute_ca_line(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["quantite", "prix_unitaire"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "quantite" in df.columns and "prix_unitaire" in df.columns:
        df["ca_ligne"] = df["quantite"].fillna(0) * df["prix_unitaire"].fillna(0)
    else:
        df["ca_ligne"] = np.nan

    return df

def compute_margin_line(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["quantite", "prix_unitaire", "cout_unitaire"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if all(col in df.columns for col in ["quantite", "prix_unitaire", "cout_unitaire"]):
        df["cout_total_ligne"] = df["quantite"].fillna(0) * df["cout_unitaire"].fillna(0)
        df["marge_ligne"] = df["ca_ligne"].fillna(0) - df["cout_total_ligne"].fillna(0)
    else:
        df["cout_total_ligne"] = np.nan
        df["marge_ligne"] = np.nan

    return df

def pareto_table(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    s = series.dropna().sort_values(ascending=False).head(top_n)
    dfp = s.reset_index()
    dfp.columns = ["label", "value"]
    dfp["cum_value"] = dfp["value"].cumsum()
    total = dfp["value"].sum()
    if total:
        dfp["cum_pct"] = dfp["cum_value"] / total * 100
        dfp["pct"] = dfp["value"] / total * 100
    else:
        dfp["cum_pct"] = 0.0
        dfp["pct"] = 0.0
    return dfp


def detect_granularity(df: pd.DataFrame, ticket_col: str | None) -> dict:
    """
    Détection plus robuste de la granularité métier.
    On ne conclut 'ligne produit' que si le ratio lignes/ticket reste plausible.
    """

    n_rows = len(df)

    if not ticket_col or ticket_col not in df.columns:
        return {
            "n_rows": n_rows,
            "n_tickets": 0,
            "ratio": None,
            "level": "non déterminable",
            "interpretation": (
                "Aucun identifiant ticket exploitable n'a été trouvé. "
                "Les KPI transactionnels ne peuvent pas être calculés de façon fiable."
            ),
            "ticket_reliable": False,
            "line_product_granularity": False,
        }

    n_tickets = int(df[ticket_col].nunique(dropna=True))
    ratio = safe_div(n_rows, n_tickets)

    if n_tickets == 0:
        return {
            "n_rows": n_rows,
            "n_tickets": 0,
            "ratio": ratio,
            "level": "non déterminable",
            "interpretation": (
                "Aucun ticket unique détecté. "
                "Les KPI transactionnels ne sont pas exploitables."
            ),
            "ticket_reliable": False,
            "line_product_granularity": False,
        }

    # Cas 1 : proche du ticket agrégé
    if ratio <= 1.2:
        return {
            "n_rows": n_rows,
            "n_tickets": n_tickets,
            "ratio": ratio,
            "level": "ticket agrégé probable",
            "interpretation": (
                "Chaque ligne semble proche d'une transaction agrégée. "
                "L'analyse détaillée produit par ticket est limitée."
            ),
            "ticket_reliable": True,
            "line_product_granularity": False,
        }

    # Cas 2 : ligne produit plausible
    if 1.2 < ratio <= 20:
        return {
            "n_rows": n_rows,
            "n_tickets": n_tickets,
            "ratio": ratio,
            "level": "ligne produit",
            "interpretation": (
                "Chaque ticket contient plusieurs lignes produit. "
                "Le ticket_id semble exploitable pour calculer le panier moyen, "
                "les KPI transactionnels et les opportunités d'upsell."
            ),
            "ticket_reliable": True,
            "line_product_granularity": True,
        }

    # Cas 3 : incohérent
    return {
        "n_rows": n_rows,
        "n_tickets": n_tickets,
        "ratio": ratio,
        "level": "granularité incohérente",
        "interpretation": (
            "Le ratio lignes / ticket est anormalement élevé. "
            "Le ticket_id ne semble pas représenter une transaction unique fiable. "
            "Les KPI transactionnels comme le panier moyen doivent être désactivés."
        ),
        "ticket_reliable": False,
        "line_product_granularity": False,
    }


def build_daily_series(df: pd.DataFrame, date_col: str = "date_vente") -> pd.Series | None:
    if date_col not in df.columns or "ca_ligne" not in df.columns:
        return None

    tmp = df.dropna(subset=[date_col, "ca_ligne"]).copy()
    if tmp.empty:
        return None

    daily = tmp.groupby(tmp[date_col].dt.date)["ca_ligne"].sum().sort_index()
    daily.index = pd.to_datetime(daily.index)
    return daily


def train_test_split_time_series(series: pd.Series, test_size: int = 14):
    if len(series) <= test_size:
        raise ValueError("Série trop courte pour un split train/test.")
    train = series.iloc[:-test_size].copy()
    test = series.iloc[-test_size:].copy()
    return train, test


def compute_forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape}


def compute_reliability_score(metrics: dict) -> dict:
    """
    Score commercial simple et lisible :
    fiabilité estimée = 100 - MAPE, bornée entre 0 et 100.
    """
    mape = metrics.get("mape", np.nan)

    if np.isnan(mape):
        return {
            "score_pct": None,
            "label": "Non calculable",
            "message": "Impossible de calculer un score de fiabilité.",
        }

    score = max(0.0, min(100.0, 100.0 - mape))

    if score >= 90:
        label = "Très fiable"
        message = "Le modèle reproduit très bien la période test."
    elif score >= 80:
        label = "Fiable"
        message = "Le modèle généralise correctement sur la période test."
    elif score >= 70:
        label = "Acceptable"
        message = "Le modèle donne une tendance exploitable, avec prudence."
    elif score >= 60:
        label = "Faible"
        message = "Le modèle reste utile pour une tendance grossière seulement."
    else:
        label = "Peu fiable"
        message = "Le modèle ne généralise pas assez bien pour inspirer confiance."

    return {"score_pct": score, "label": label, "message": message}


def interpret_generalization(metrics: dict) -> dict:
    mape = metrics.get("mape", np.nan)

    if np.isnan(mape):
        return {
            "label": "Interprétation impossible",
            "message": "MAPE non calculable.",
            "ok": False,
        }

    if mape < 10:
        return {
            "label": "Bonne généralisation",
            "message": "Le modèle prédit bien les données non vues.",
            "ok": True,
        }
    elif mape < 20:
        return {
            "label": "Généralisation correcte",
            "message": "Le modèle est exploitable pour une tendance, avec prudence.",
            "ok": True,
        }
    elif mape < 30:
        return {
            "label": "Généralisation moyenne",
            "message": "Le modèle capte une partie de la dynamique, mais l’erreur reste élevée.",
            "ok": False,
        }
    else:
        return {
            "label": "Mauvaise généralisation",
            "message": "Le modèle ne généralise pas correctement.",
            "ok": False,
        }


def validate_holt_winters(series: pd.Series, test_size: int = 14, seasonal_periods: int = 7):
    if not STATSMODELS_OK:
        return None, "statsmodels n'est pas installé."

    min_required = max(30, seasonal_periods * 2 + test_size)
    if len(series) < min_required:
        return None, f"Série trop courte. Minimum recommandé : {min_required} points."

    try:
        train, test = train_test_split_time_series(series, test_size=test_size)

        model = ExponentialSmoothing(
            train.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        )
        fit = model.fit(optimized=True)

        pred = fit.forecast(len(test))
        pred.index = test.index

        metrics = compute_forecast_metrics(test, pred)
        interpretation = interpret_generalization(metrics)
        reliability = compute_reliability_score(metrics)

        params = fit.params if hasattr(fit, "params") else {}

        return {
            "train": train,
            "test": test,
            "pred": pred,
            "metrics": metrics,
            "interpretation": interpretation,
            "reliability": reliability,
            "alpha": params.get("smoothing_level"),
            "beta": params.get("smoothing_trend"),
            "gamma": params.get("smoothing_seasonal"),
        }, None

    except Exception as e:
        return None, str(e)


def build_commercial_insights(
    df: pd.DataFrame,
    date_col: str | None,
    ticket_col: str | None,
    prod_col: str | None,
    cat_col: str | None,
    pay_col: str | None,
) -> tuple[list[str], list[str]]:
    insights = []
    risks = []

    if "ca_ligne" not in df.columns or df["ca_ligne"].dropna().empty:
        return [], ["Impossible de produire des insights commerciaux : chiffre d’affaires indisponible."]

    # 1. Produit n°1 en CA
    if prod_col and prod_col in df.columns:
        prod_ca = (
            df.groupby(prod_col, dropna=False)["ca_ligne"]
            .sum()
            .sort_values(ascending=False)
        )
        if not prod_ca.empty:
            top_product = prod_ca.index[0]
            top_product_ca = prod_ca.iloc[0]
            insights.append(
                f"💰 Le produit qui génère le plus de chiffre d’affaires est **{top_product}** avec **{euro(top_product_ca)}**."
            )

    # 2. Produits rentables / peu rentables si marge dispo
    if (
        prod_col
        and prod_col in df.columns
        and "marge_ligne" in df.columns
        and df["marge_ligne"].notna().any()
    ):
        perf = (
            df.groupby(prod_col, dropna=False)
            .agg(
                ca=("ca_ligne", "sum"),
                marge=("marge_ligne", "sum"),
            )
            .reset_index()
        )

        perf["taux_marge"] = np.where(perf["ca"] > 0, perf["marge"] / perf["ca"], np.nan)

        best_margin = perf.sort_values("marge", ascending=False).iloc[0]
        low_margin = perf.sort_values("taux_marge", ascending=True).iloc[0]
        hidden = perf.sort_values(["taux_marge", "marge"], ascending=[False, False]).iloc[0]

        insights.append(
            f"📈 Le produit le plus rentable en marge est **{best_margin[prod_col]}** avec **{euro(best_margin['marge'])}**."
        )

        insights.append(
            f"🎯 Le produit à fort potentiel à mettre davantage en avant semble être **{hidden[prod_col]}**."
        )

        risks.append(
            f"⚠️ Le produit le moins rentable en taux de marge est **{low_margin[prod_col]}**. Il mérite d’être revu ou moins poussé."
        )

    # 3. Panier moyen uniquement si ticket fiable
    if ticket_col and ticket_col in df.columns:
        n_rows = len(df)
        n_tickets = df[ticket_col].nunique(dropna=True)
        ratio = safe_div(n_rows, n_tickets)

        if 1.2 < ratio <= 20:
            panier = df.groupby(ticket_col)["ca_ligne"].sum()
            if not panier.empty:
                panier_moyen = panier.mean()
                insights.append(
                    f"🧾 Le panier moyen actuel est de **{euro(panier_moyen)}**. Une légère hausse peut générer un gain significatif à l’échelle du mois."
                )
        else:
            risks.append(
                "⚠️ Le panier moyen n’est pas affiché car l’identifiant ticket ne semble pas assez fiable pour une analyse transactionnelle."
            )

    # 4. Heure rentable uniquement si vraie heure exploitable
    if "heure_num" in df.columns and df["heure_num"].dropna().nunique() > 1:
        metric_col = "marge_ligne" if "marge_ligne" in df.columns and df["marge_ligne"].notna().any() else "ca_ligne"
        by_hour = df.groupby("heure_num")[metric_col].sum().sort_values(ascending=False)

        if not by_hour.empty:
            best_hour = int(by_hour.index[0])
            insights.append(
                f"⏰ Le créneau le plus performant semble être **{best_hour}h**. C’est un horaire à exploiter commercialement."
            )

    # 5. Paiement principal
    if pay_col and pay_col in df.columns:
        pay_counts = df[pay_col].astype(str).value_counts(dropna=False)
        if not pay_counts.empty:
            main_pay = pay_counts.index[0]
            insights.append(
                f"💳 Le mode de paiement le plus fréquent est **{main_pay}**."
            )

    # 6. Catégorie n°1
    if cat_col and cat_col in df.columns:
        cat_ca = df.groupby(cat_col)["ca_ligne"].sum().sort_values(ascending=False)
        if not cat_ca.empty:
            best_cat = cat_ca.index[0]
            insights.append(
                f"🏷️ La catégorie la plus contributrice au chiffre d’affaires est **{best_cat}**."
            )

    if not insights:
        insights.append(
            "Les données sont lisibles, mais la structure disponible limite les recommandations automatiques."
        )

    return insights, risks

   



def simulate_gain(df: pd.DataFrame, ticket_col: str | None, date_col: str | None, uplift_panier: float = 1.0):
    if not ticket_col or ticket_col not in df.columns or not date_col or date_col not in df.columns:
        return {"gain_journalier": None, "gain_mensuel": None, "gain_annuel": None}

    n_rows = len(df)
    n_tickets = df[ticket_col].nunique(dropna=True)
    ratio = safe_div(n_rows, n_tickets)

    # On bloque la simulation si le ticket n'est pas fiable
    if not (1.2 < ratio <= 20):
        return {"gain_journalier": None, "gain_mensuel": None, "gain_annuel": None}

    tmp = df.dropna(subset=[date_col]).copy()
    if tmp.empty:
        return {"gain_journalier": None, "gain_mensuel": None, "gain_annuel": None}

    tickets_per_day = tmp.groupby(tmp[date_col].dt.date)[ticket_col].nunique().mean()
    gain_journalier = tickets_per_day * uplift_panier
    gain_mensuel = gain_journalier * 30
    gain_annuel = gain_journalier * 365

    return {
        "gain_journalier": gain_journalier,
        "gain_mensuel": gain_mensuel,
        "gain_annuel": gain_annuel,
    }
# =========================================================
# CHARGEMENT / SOURCE
# =========================================================
# 🔥 SOURCE UNIQUE (PROPRE)
# =========================================================
# CHARGEMENT / SOURCE
# =========================================================
data_path = Path(r"C:\Users\lyesm\OneDrive\Desktop\ventes_60j.csv")

df = pd.read_csv(data_path)
summary = {}

DATE_COL = "date_vente" if "date_vente" in df.columns else None
TIME_COL = "heure" if "heure" in df.columns else None
TICKET_COL = "ticket_id" if "ticket_id" in df.columns else None
CAT_COL = "categorie" if "categorie" in df.columns else None
PROD_COL = "produit" if "produit" in df.columns else None
PAY_COL = "moyen_paiement" if "moyen_paiement" in df.columns else None

if DATE_COL and TIME_COL:
    df["datetime"] = pd.to_datetime(
        df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
        errors="coerce"
    )
    df[DATE_COL] = df["datetime"]
    df["heure_num"] = df["datetime"].dt.hour
else:
    df["datetime"] = pd.NaT
    df["heure_num"] = np.nan

df = compute_ca_line(df)
df = compute_margin_line(df)

granularity = detect_granularity(df, TICKET_COL)
# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="main-title">Augmentez votre chiffre d’affaires grâce à vos données 📊</div>
        <div class="subtitle">
            Ce tableau de bord transforme un simple export de caisse en décisions concrètes :
            produits à pousser, heures fortes, opportunités de vente, et estimation de fiabilité des prévisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FILTRES
# =========================================================
with st.sidebar:
    st.header("🔎 Filtres")
    df_f = df.copy()

    if DATE_COL and df_f[DATE_COL].notna().any():
        dmin = df_f[DATE_COL].min().date()
        dmax = df_f[DATE_COL].max().date()
        date_range = st.date_input(
            "Période",
            value=(dmin, dmax),
            min_value=dmin,
            max_value=dmax,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            df_f = df_f[
                (df_f[DATE_COL].dt.date >= start)
                & (df_f[DATE_COL].dt.date <= end)
            ]

    if CAT_COL:
        cats = sorted(df_f[CAT_COL].dropna().astype(str).unique().tolist())
        selected_cats = st.multiselect("Catégories", options=cats, default=cats)
        if selected_cats:
            df_f = df_f[df_f[CAT_COL].astype(str).isin(selected_cats)]

    if PAY_COL:
        pays = sorted(df_f[PAY_COL].dropna().astype(str).unique().tolist())
        selected_pay = st.multiselect("Modes de paiement", options=pays, default=pays)
        if selected_pay:
            df_f = df_f[df_f[PAY_COL].astype(str).isin(selected_pay)]

# Série journalière pour prévision
daily_series = None
if DATE_COL and "ca_ligne" in df_f.columns:
    daily_series = build_daily_series(df_f, DATE_COL)

# =========================================================
# KPI
# =========================================================
nb_rows = len(df_f)
nb_tickets = int(df_f[TICKET_COL].nunique(dropna=True)) if TICKET_COL else None
ca_total = float(df_f["ca_ligne"].dropna().sum()) if "ca_ligne" in df_f.columns else None
panier_moyen = (
    ca_total / nb_tickets
    if (
        ca_total is not None
        and nb_tickets
        and nb_tickets > 0
        and granularity["ticket_reliable"]
    )
    else None
)
marge_total = (
    float(df_f["marge_ligne"].dropna().sum())
    if "marge_ligne" in df_f.columns and df_f["marge_ligne"].notna().any()
    else None
)
taux_marge = (
    safe_div(marge_total, ca_total) * 100
    if marge_total is not None and ca_total not in (None, 0)
    else None
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Lignes", f"{nb_rows:,}".replace(",", " "))
k2.metric("Tickets", f"{nb_tickets:,}".replace(",", " ") if nb_tickets is not None else "N/A")
k3.metric("CA total", euro(ca_total))
if granularity["ticket_reliable"]:
    k4.metric("Panier moyen", euro(panier_moyen))
else:
    k4.metric("Panier moyen", "⚠️ Donnée non exploitable")
k5.metric("Taux de marge", pct(taux_marge))

if DATE_COL and df_f[DATE_COL].notna().any():
    dmin = df_f[DATE_COL].min()
    dmax = df_f[DATE_COL].max()
    st.caption(f"📅 Période analysée : {dmin.date()} → {dmax.date()}")

# =========================================================
# INSIGHTS COMMERCIAUX
# =========================================================
insights, risks = build_commercial_insights(df_f, DATE_COL, TICKET_COL, PROD_COL, CAT_COL, PAY_COL)

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Ce que vos données racontent")
    for item in insights:
        st.markdown(f'<div class="insight-box">{item}</div>', unsafe_allow_html=True)

    for item in risks:
        st.markdown(f'<div class="risk-box">{item}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="cta-box">', unsafe_allow_html=True)
    st.subheader("Simulation de gain")
    uplift = st.slider("Hausse du panier moyen simulée (€)", 0.5, 5.0, 1.0, 0.5)
    sim = simulate_gain(df_f, TICKET_COL, DATE_COL, uplift_panier=uplift)

    st.metric("Gain journalier estimé", euro(sim["gain_journalier"]))
    st.metric("Gain mensuel estimé", euro(sim["gain_mensuel"]))
    st.metric("Gain annuel estimé", euro(sim["gain_annuel"]))

    st.caption(
        "Hypothèse simple : même volume de tickets, panier moyen légèrement plus élevé."
    )
    st.markdown('</div>', unsafe_allow_html=True)

ratio_text = f"{granularity['ratio']:.2f}" if granularity["ratio"] is not None else "N/A"

if granularity["ticket_reliable"]:
    st.success(
        f"Granularité détectée : **{granularity['level']}** — "
        f"{granularity['interpretation']} "
        f"(ratio lignes / ticket : **{ratio_text}**)."
    )
else:
    st.warning(
        f"Granularité détectée : **{granularity['level']}** — "
        f"{granularity['interpretation']} "
        f"(ratio lignes / ticket : **{ratio_text}**)."
    )



# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Activité",
        "📌 Produits & catégories",
        "💳 Paiements",
        "🔮 Prévision & fiabilité",
        "🧾 Données",
    ]
)

# =========================================================
# TAB 1 - ACTIVITE
# =========================================================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("CA journalier")
        if DATE_COL and "ca_ligne" in df_f.columns and df_f[DATE_COL].notna().any():
            daily = (
                df_f.dropna(subset=[DATE_COL])
                .groupby(df_f[DATE_COL].dt.date)["ca_ligne"]
                .sum()
                .reset_index()
            )
            daily.columns = ["date", "ca"]
            daily["tendance_7j"] = daily["ca"].rolling(7).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["date"], y=daily["ca"], mode="lines+markers", name="CA"))
            fig.add_trace(
                go.Scatter(
                    x=daily["date"],
                    y=daily["tendance_7j"],
                    mode="lines",
                    name="Tendance 7j",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(height=420, xaxis_title="Date", yaxis_title="CA")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Impossible de tracer le CA journalier.")

    with c2:
        st.subheader("CA mensuel")
        if DATE_COL and "ca_ligne" in df_f.columns and df_f[DATE_COL].notna().any():
            tmp = df_f.dropna(subset=[DATE_COL]).copy()
            tmp["month"] = tmp[DATE_COL].dt.to_period("M").astype(str)
            monthly = tmp.groupby("month")["ca_ligne"].sum().reset_index()
            fig = px.bar(monthly, x="month", y="ca_ligne")
            fig.update_layout(height=420, xaxis_title="Mois", yaxis_title="CA")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Impossible de tracer le CA mensuel.")

# =========================================================
# TAB 2 - PRODUITS
# =========================================================
with tab2:
    a, b = st.columns(2)

    with a:
        st.subheader("Pareto produits (CA)")
        if PROD_COL and "ca_ligne" in df_f.columns:
            prod_ca = (
                df_f.groupby(PROD_COL, dropna=False)["ca_ligne"]
                .sum()
                .sort_values(ascending=False)
            )
            ptab = pareto_table(prod_ca, top_n=20)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ptab["label"], y=ptab["value"], name="CA"))
            fig.add_trace(
                go.Scatter(
                    x=ptab["label"],
                    y=ptab["cum_pct"],
                    mode="lines+markers",
                    name="% cumulé",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=450,
                xaxis_title="Produit",
                yaxis_title="CA",
                yaxis2=dict(title="% cumulé", overlaying="y", side="right", range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ptab, use_container_width=True)
        else:
            st.info("Impossible : colonne produit ou CA manquant.")

    with b:
        st.subheader("Pareto catégories (CA)")
        if CAT_COL and "ca_ligne" in df_f.columns:
            cat_ca = (
                df_f.groupby(CAT_COL, dropna=False)["ca_ligne"]
                .sum()
                .sort_values(ascending=False)
            )
            ctab = pareto_table(cat_ca, top_n=10)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ctab["label"], y=ctab["value"], name="CA"))
            fig.add_trace(
                go.Scatter(
                    x=ctab["label"],
                    y=ctab["cum_pct"],
                    mode="lines+markers",
                    name="% cumulé",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=450,
                xaxis_title="Catégorie",
                yaxis_title="CA",
                yaxis2=dict(title="% cumulé", overlaying="y", side="right", range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ctab, use_container_width=True)
        else:
            st.info("Impossible : catégorie ou CA manquant.")

    if PROD_COL and "ca_ligne" in df_f.columns:
        st.subheader("Top produits")
        agg_cols = {"ca_ligne": "sum"}
        if "marge_ligne" in df_f.columns and df_f["marge_ligne"].notna().any():
            agg_cols["marge_ligne"] = "sum"
        if "quantite" in df_f.columns:
            agg_cols["quantite"] = "sum"

        top_products = (
            df_f.groupby(PROD_COL, dropna=False)
            .agg(agg_cols)
            .reset_index()
            .sort_values("ca_ligne", ascending=False)
        )
        st.dataframe(top_products.head(15), use_container_width=True)

# =========================================================
# TAB 3 - PAIEMENTS
# =========================================================
with tab3:
    st.subheader("Répartition des paiements")

    if PAY_COL:
        pcount = df_f[PAY_COL].astype(str).value_counts(dropna=False).reset_index()
        pcount.columns = ["mode_paiement", "nb_lignes"]

        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(pcount, names="mode_paiement", values="nb_lignes")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if TICKET_COL and "ca_ligne" in df_f.columns:
                tmp = (
                    df_f.groupby([TICKET_COL, PAY_COL], dropna=False)["ca_ligne"]
                    .sum()
                    .reset_index()
                )
                panier = tmp.groupby(PAY_COL)["ca_ligne"].mean().reset_index()
                panier.columns = [PAY_COL, "panier_moyen"]
                fig2 = px.bar(panier, x=PAY_COL, y="panier_moyen")
                fig2.update_layout(height=420, yaxis_title="Panier moyen")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Colonne mode_paiement absente.")

# =========================================================
# TAB 4 - PREVISION & FIABILITE
# =========================================================
with tab4:
    st.subheader("Prévision du chiffre d’affaires")
    st.caption(
        "La prévision n’est utile commercialement que si elle tient correctement sur une période test non vue."
    )

    if daily_series is None or len(daily_series) < 30:
        st.warning("Pas assez de données journalières pour tester sérieusement une prévision.")
    else:
        st.write(f"Nombre de jours disponibles : **{len(daily_series)}**")

        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Taille de la période test (jours)", 7, 21, 14, 1)
        with c2:
            seasonal_periods = st.selectbox("Saisonnalité hebdomadaire", options=[7], index=0)

        validation_result, validation_error = validate_holt_winters(
            daily_series,
            test_size=test_size,
            seasonal_periods=seasonal_periods,
        )

        # Historique brut
        hist_df = pd.DataFrame(
            {"date": daily_series.index, "ca": daily_series.values}
        )
        hist_df["tendance_7j"] = hist_df["ca"].rolling(7).mean()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["ca"], mode="lines+markers", name="Historique"))
        fig_hist.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["tendance_7j"],
                mode="lines",
                name="Tendance 7j",
                line=dict(dash="dash"),
            )
        )
        fig_hist.update_layout(height=420, xaxis_title="Date", yaxis_title="CA")
        st.plotly_chart(fig_hist, use_container_width=True)

        if validation_error:
            st.error(f"Validation impossible : {validation_error}")
        else:
            train = validation_result["train"]
            test = validation_result["test"]
            pred = validation_result["pred"]
            metrics = validation_result["metrics"]
            interp = validation_result["interpretation"]
            reliability = validation_result["reliability"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{metrics['mae']:.2f}")
            m2.metric("RMSE", f"{metrics['rmse']:.2f}")
            m3.metric("MAPE", f"{metrics['mape']:.2f} %")
            m4.metric(
                "Fiabilité estimée",
                f"{reliability['score_pct']:.1f} %" if reliability["score_pct"] is not None else "N/A",
            )

            p1, p2, p3 = st.columns(3)
            p1.metric("Alpha", f"{validation_result['alpha']:.3f}" if validation_result["alpha"] is not None else "N/A")
            p2.metric("Beta", f"{validation_result['beta']:.3f}" if validation_result["beta"] is not None else "N/A")
            p3.metric("Gamma", f"{validation_result['gamma']:.3f}" if validation_result["gamma"] is not None else "N/A")

            st.write(f"Période train : **{len(train)} jours**")
            st.write(f"Période test : **{len(test)} jours**")

            if reliability["score_pct"] is not None:
                if reliability["score_pct"] >= 80:
                    st.success(
                        f"**{reliability['label']}** — {reliability['message']} "
                        f"(score estimé : {reliability['score_pct']:.1f} %)."
                    )
                elif reliability["score_pct"] >= 70:
                    st.info(
                        f"**{reliability['label']}** — {reliability['message']} "
                        f"(score estimé : {reliability['score_pct']:.1f} %)."
                    )
                else:
                    st.warning(
                        f"**{reliability['label']}** — {reliability['message']} "
                        f"(score estimé : {reliability['score_pct']:.1f} %)."
                    )

            compare_df = pd.DataFrame(
                {
                    "date": list(train.index) + list(test.index) + list(pred.index),
                    "valeur": list(train.values) + list(test.values) + list(pred.values),
                    "serie": (
                        ["Train"] * len(train)
                        + ["Test réel"] * len(test)
                        + ["Prévision test"] * len(pred)
                    ),
                }
            )

            fig = px.line(
                compare_df,
                x="date",
                y="valeur",
                color="serie",
                markers=True,
                title="Validation chronologique : train / test / prévision",
            )
            fig.update_layout(height=460, xaxis_title="Date", yaxis_title="CA")
            st.plotly_chart(fig, use_container_width=True)

            details_df = pd.DataFrame(
                {
                    "date": test.index,
                    "ca_reel": test.values,
                    "ca_prevu": pred.values,
                    "erreur_absolue": np.abs(test.values - pred.values),
                }
            )
            st.dataframe(details_df, use_container_width=True)

            st.markdown("### Lecture commerciale")
            if reliability["score_pct"] is not None and reliability["score_pct"] >= 80:
                st.success(
                    "La prévision peut être utilisée pour illustrer une tendance crédible. "
                    "C’est utile pour anticiper l’activité et mieux préparer les décisions commerciales."
                )
            elif reliability["score_pct"] is not None and reliability["score_pct"] >= 70:
                st.info(
                    "La prévision donne une tendance exploitable, mais doit être présentée avec prudence."
                )
            else:
                st.error(
                    "La prévision ne doit pas être vendue comme fiable en l’état. "
                    "Il faut plus d’historique ou un autre modèle."
                )

# =========================================================
# TAB 5 - DONNEES
# =========================================================
with tab5:
    st.subheader("Aperçu des données analysées")

    st.markdown("**Source des données**")
    st.code(str(data_path))

    st.markdown("**Extrait des ventes**")
    st.dataframe(df_f.head(100), use_container_width=True)