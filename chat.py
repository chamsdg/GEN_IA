# chat.py
import streamlit as st
import pandas as pd
import time

from fonction import clean_html_tags, markdown_to_html,find_clients_in_question
from llm_runner import run_llm

# 🔹 NOUVEAUX IMPORTS

from intent import is_evolution_question, is_comparison_question
from ventes import build_monthly_sales, build_monthly_sales_by_client
from charts import build_line_chart, build_multi_line_chart


 

def normalize_client(x: str) -> str:
    return (
        x.lower()
        .replace(" sa", "")
        .replace(" sarl", "")
        .replace(" ltd", "")
        .replace(".", "")
        .strip()
    )


# ============================================================
# INIT STATE
# ============================================================
def init_chat_state():
    defaults = {
        "history": [],
        "pending_question": None,
        "suggestion_input": None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ============================================================
# HISTORY
# ============================================================
def render_history():
    for entry in st.session_state.history:
        with st.chat_message("user", avatar="🧑🏿‍💼"):
            st.markdown(f"🕒 {entry['time']}  \n**{entry['question']}**")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(entry["answer"])

# ============================================================
# INPUT
# ============================================================
def handle_user_input():
    user_input = st.chat_input("Demandez à GEN'AI CVM...")

    if st.session_state.suggestion_input:
        user_input = st.session_state.suggestion_input
        st.session_state.suggestion_input = None

    if user_input:
        st.session_state.pending_question = user_input

    return st.session_state.pending_question



# ============================================================
# PROCESS QUESTION
# ============================================================
def process_question(model, temperature):
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    # ---- USER MESSAGE
    with st.chat_message("user", avatar="🧑🏿‍💼"):
        st.markdown(question)

    # ---- ASSISTANT
    with st.chat_message("assistant", avatar="🤖"):
        progress = st.progress(0)

        # 1️⃣ Réponse LLM (texte / tableau)
        answer, duration = run_llm(
            question=question,
            model=model,
            temperature=temperature,
            progress=progress
        )

        st.markdown(answer, unsafe_allow_html=True)
        st.markdown(f"⏱️ **Temps de réponse : {duration:.2f} s**")

        # ====================================================
        # 2️⃣ SI QUESTION = ÉVOLUTION → GRAPH INTELLIGENT
        # ====================================================
        if is_evolution_question(question):

            # 🔹 Récupération des données de ventes
            fact = st.session_state.get("fact", pd.DataFrame())

            if not fact.empty and "RAISON_SOCIALE" in fact.columns:

                # 🔹 Liste clients disponibles
                client_list = (
                    fact["RAISON_SOCIALE"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                # 🔍 Détection fuzzy des clients (SNIM, etc.)
                clients_detectes = find_clients_in_question(
                    question=question,
                    client_list=client_list
                )

                # 🔹 Filtrage selon le contexte détecté
                if len(clients_detectes) == 1:
                    fact_filtered = fact[
                        fact["RAISON_SOCIALE"] == clients_detectes[0]
                    ]
                    title = f"📈 Évolution mensuelle des ventes – {clients_detectes[0]}"

                elif len(clients_detectes) > 1:
                    fact_filtered = fact[
                        fact["RAISON_SOCIALE"].isin(clients_detectes)
                    ]
                    title = "📈 Évolution mensuelle des ventes – Comparaison clients"

                else:
                    fact_filtered = fact
                    title = "📈 Évolution mensuelle des ventes (global)"

                # 🔹 Construction des données mensuelles
                df_monthly = build_monthly_sales(fact_filtered)

                # 🔹 Tracé du graphique
                if not df_monthly.empty:
                    fig = build_line_chart(
                        df=df_monthly,
                        x_col="ANNEE_MOIS",
                        y_col="total_sales",
                        title=title,
                        y_label="CA (€)"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # ====================================================
        # 3️⃣ Historique
        # ====================================================
        st.session_state.history.append({
            "time": time.strftime("%H:%M:%S"),
            "question": question,
            "answer": answer
        })



# ============================================================
# SEARCH
# ============================================================
def render_search():
    with st.sidebar:
        st.markdown("### 🔍 Rechercher dans l'historique")
        query = st.text_input("Mot-clé")

        if query:
            matches = [
                h for h in st.session_state.history
                if query.lower() in h["question"].lower()
                or query.lower() in h["answer"].lower()
            ]

            if matches:
                for m in matches:
                    with st.expander(f"{m['time']} | {m['question']}"):
                        st.markdown(
                            clean_html_tags(markdown_to_html(m["answer"])),
                            unsafe_allow_html=True
                        )
            else:
                st.info("Aucun résultat.")
