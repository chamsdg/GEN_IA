#sidebar.py

import streamlit as st
import pandas as pd
from fonction import load_data_from_snowflake


def render_sidebar():
    with st.sidebar:
        st.subheader("Informations système")

        try:
            data = load_data_from_snowflake()

            fact = data.get("fact", pd.DataFrame())
            final = data.get("final", pd.DataFrame())
            opportunite_pays = data.get("opportunite_pays", pd.DataFrame())
            opportunite_bu = data.get("opportunite_bu", pd.DataFrame())
            equipement = data.get("equipement", pd.DataFrame())

            # ================================
            # NORMALISATION DATE FACTURE
            # ================================
            if not fact.empty and "DATE_FACTURE" in fact.columns:
                fact["date_facture_dt"] = pd.to_datetime(
                    fact["DATE_FACTURE"],
                    errors="coerce",
                    dayfirst=True
                )

            # ================================
            # SESSION STATE = SOURCE DE VÉRITÉ
            # ================================
            st.session_state.fact = fact
            st.session_state.final = final
            st.session_state.opportunite_pays = opportunite_pays
            st.session_state.opportunite_bu = opportunite_bu
            st.session_state.equipement = equipement

            st.success("Données Snowflake chargées (cache actif)")

            if fact.empty:
                st.warning("Factures vides")
            else:
                st.caption(f"📊 Factures : {len(fact)} lignes")
                st.caption(
                    f"📅 Dates valides : {fact['date_facture_dt'].notna().sum()}"
                )

            if final.empty:
                st.warning("Opportunités vides")

        except Exception as e:
            st.error("Erreur lors du chargement Snowflake")
            st.error(str(e))

        st.markdown("---")
        st.markdown("### Paramètres")

        if st.button("🧹 Effacer l’historique"):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
