# chat.py
import streamlit as st
import pandas as pd
import time
import re

from fonction import clean_html_tags, markdown_to_html,find_clients_in_question
from llm_runner import run_llm

# 🔹 NOUVEAUX IMPORTS

from intent import is_evolution_question, is_comparison_question
from ventes import build_monthly_sales, build_monthly_sales_by_client
from charts import build_line_chart, build_multi_line_chart
from charts import build_evolution_title





def detect_clients_for_comparison(
    question: str,
    client_list: list,
    max_clients: int = 2
) -> list:
    

    q_lower = question.lower()

    # 1️⃣ clients explicitement cités (PRIORITÉ ABSOLUE)
    explicit_clients = [
        client for client in client_list
        if client.lower() in q_lower
    ]

    # 2️⃣ fallback : fuzzy global (fonction existante)
    fuzzy_clients = find_clients_in_question(question, client_list)

    # 3️⃣ union en respectant la priorité
    ordered_clients = []

    for client in explicit_clients:
        if client not in ordered_clients:
            ordered_clients.append(client)

    for client in fuzzy_clients:
        if client not in ordered_clients:
            ordered_clients.append(client)

    # 4️⃣ suppression des clients parents (groupe vs filiale)
    def remove_parent_clients(clients):
        result = []
        for c in clients:
            if not any(
                c != other and c.lower() in other.lower()
                for other in clients
            ):
                result.append(c)
        return result

    ordered_clients = remove_parent_clients(ordered_clients)

    # 5️⃣ limite stricte
    return ordered_clients[:max_clients]



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
            st.markdown(
                f"""
                <div style='background-color:#1E90FF; color:white;
                padding:1rem; border-radius:8px;'>
                {entry['answer']}
                </div>
                """,
                unsafe_allow_html=True
            )


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
       
        
        # Sécurité : copie
        fact = fact.copy()
        
        # 🔹 Conversion robuste de la date
        fact["date_facture_dt"] = pd.to_datetime(
        fact["DATE_FACTURE"], errors="coerce"
    )


    
        if not fact.empty and "RAISON_SOCIALE" in fact.columns:
    
            # 🔹 Liste clients disponibles
            client_list = (
                fact["RAISON_SOCIALE"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
    
            # 🔍 Détection fuzzy des clients
            # 🔍 Détection fuzzy des clients
            # 🔍 Détection fuzzy des clients
           # clients_detectes = find_clients_in_question(
           #     question=question,
          #      client_list=client_list
           # )
            clients_detectes = detect_clients_for_comparison(question, client_list)
            
            # ✅ FORÇAGE MONO-CLIENT SI NOM EXPLICITE DANS LA QUESTION
            question_lower = question.lower()
            # 🔍 Détection fuzzy des clients
            
            
            question_lower = question.lower()
            
            # 🔒 Détection explicite de comparaison
            comparison_markers = [" vs ", " et ", " entre ", "compar", " versus "]
            is_comparison = any(m in question_lower for m in comparison_markers)
            
            # ✅ Forcer mono-client UNIQUEMENT s'il n'y a PAS de comparaison
            if not is_comparison:
                for client in client_list:
                    if client.lower() in question_lower:
                        clients_detectes = [client]
                        break

            """
            for client in client_list:
                if client.lower() in question_lower:
                    clients_detectes = [client]
                    break

            
            clients_detectes = find_clients_in_question(
                question=question,
                client_list=client_list
            )
            """
    
            # 🔹 Filtrage selon le contexte détecté
            # 🔹 Filtrage des données
            # 🔹 Filtrage selon le contexte détecté
            #if len(clients_detectes) == 1:
            # 🔹 Détection & filtrage des clients
            if len(clients_detectes) == 1:
                client = clients_detectes[0]
            
                fact_filtered = fact[
                    fact["RAISON_SOCIALE"] == client
                ]
                group_by_client = False
            
            elif len(clients_detectes) > 1:
                fact_filtered = fact[
                    fact["RAISON_SOCIALE"].isin(clients_detectes)
                ]
                group_by_client = True
            
            else:
                fact_filtered = fact
                group_by_client = False
            
            
            # 🔹 Construction du titre (UNE seule fois)
            title = build_evolution_title(clients_detectes)
            
            
            # 🔹 Construction des données mensuelles
            df_monthly = build_monthly_sales(
                fact_filtered,
                group_by_client=group_by_client
            )
            
            
            # 🔹 Tracé du graphique
            if not df_monthly.empty:
            
                if group_by_client and "RAISON_SOCIALE" in df_monthly.columns:
                    fig = build_multi_line_chart(
                        df=df_monthly,
                        x_col="ANNEE_MOIS",
                        y_col="total_sales",
                        color_col="RAISON_SOCIALE",
                        title=title,
                        y_label="CA (€)"
                    )
                else:
                    fig = build_line_chart(
                        df=df_monthly,
                        x_col="ANNEE_MOIS",
                        y_col="total_sales",
                        title=title,
                        y_label="CA (€)"
                    )
            
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Aucune donnée disponible pour cette requête.")




    
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
