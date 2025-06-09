import streamlit as st
import joblib
import pandas as pd
import datetime

# Charger modèle et colonnes
model = joblib.load("xgb_home_credit_pipeline.joblib")
numeric_cols, categorical_cols = joblib.load("features_used.joblib")
all_cols = numeric_cols + categorical_cols

st.set_page_config(page_title="Scoring client - Néo Banque", layout="centered")

st.title("📊 Évaluation du risque client")

# Formulaire Streamlit
with st.form("formulaire"):
    nom = st.text_input("Nom", value="Dupont")
    prenom = st.text_input("Prénom", value="Jean")
    naissance = st.date_input("Date de naissance", value=datetime.date(1980, 1, 1))
    statut = st.text_input("Statut", value="Cadre")
    revenus = st.number_input("Revenus (€)", value=4500)
    credit = st.number_input("Montant du crédit demandé (€)", value=100000)
    adresse = st.text_input("Adresse", value="123 Rue Exemple")
    submit = st.form_submit_button("Évaluer le client")

if submit:
    # Préparer les données
    input_data = {col: 0.0 for col in all_cols}
    input_data["AMT_INCOME_TOTAL"] = revenus
    input_data["AMT_CREDIT"] = credit
    input_data["DAYS_BIRTH"] = -((pd.Timestamp.today() - pd.to_datetime(naissance)).days)

    df = pd.DataFrame([input_data])
    proba = model.predict_proba(df)[0][1]
    decision = "✅ Éligible" if proba < 0.4 else "⚠️ Risque modéré" if proba < 0.7 else "❌ Risque élevé"

    st.metric("Score prédictif", f"{proba:.2%}")
    st.success(f"Décision : {decision}")
