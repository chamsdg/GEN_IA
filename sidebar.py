# sidebar.py
import streamlit as st
import pandas as pd
from datetime import datetime
from fonction import load_data_from_snowflake


def render_sidebar():
    with st.sidebar:
        st.subheader("Informations système")

        # ================================
        # BOUTON RAFRAÎCHISSEMENT MANUEL
        # ================================
        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.session_state.last_refresh = None
            st.rerun()

        try:
            # ================================
            # CHARGEMENT DONNÉES (CACHE)
            # ================================
            data = load_data_from_snowflake()

            # ================================
            # DATE DE DERNIER RAFRAÎCHISSEMENT
            # ================================
            if "last_refresh" not in st.session_state or st.session_state.last_refresh is None:
                st.session_state.last_refresh = datetime.now()

            st.caption(
                f"🕒 Données mises à jour le "
                f"{st.session_state.last_refresh.strftime('%d/%m/%Y à %H:%M:%S')}"
            )

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

            # ================================
            # STATUT DES TABLES
            # ================================
            if fact.empty:
                st.warning("Factures vides")
            else:
                st.caption(f"📊 Factures : {len(fact)} lignes")

            if opportunite_pays.empty:
                st.warning("Opportunités pays vides")
            else:
                st.caption(f"📊 Opportunités pays : {len(opportunite_pays)} lignes")

            if opportunite_bu.empty:
                st.warning("Opportunités BU vides")
            else:
                st.caption(f"📊 Opportunités BU : {len(opportunite_bu)} lignes")

            if equipement.empty:
                st.warning("Équipements vides")
            else:
                st.caption(f"📊 Équipements : {len(equipement)} lignes")

        except Exception as e:
            st.error("Erreur lors du chargement Snowflake")
            st.error(str(e))

        st.markdown("---")
        st.markdown("### Paramètres")

        if st.button("🧹 Effacer l’historique"):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
