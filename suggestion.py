# suggestions.py
import streamlit as st


SUGGESTIONS = [
    {
        "key": "division",
        "icon": "🧩",
        "title": "Répartition des ventes",
        "subtitle": "Par division",
        "value": "Peux-tu me donner la répartition des ventes par division ?"
    },
    {
        "key": "constructeur",
        "icon": "🏭",
        "title": "Top constructeur",
        "subtitle": "Ventes par constructeur",
        "value": "Analyse et donne les ventes par constructeur"
    },
    {
        "key": "neemba",
        "icon": "🏢",
        "title": "Présentation",
        "subtitle": "Groupe Neemba",
        "value": "Qu'est-ce que tu peux me dire sur Neemba ?"
    },
    {
        "key": "major",
        "icon": "🧰",
        "title": "Major Classe SNIM",
        "subtitle": "Produits les plus consommés",
        "value": "Donne le top 3 des major classe et produits les plus consommés par la SNIM"
    },
    {
        "key": "ca_year",
        "icon": "📈",
        "title": "Chiffre d'affaires",
        "subtitle": "Année en cours",
        "value": "Quel est le chiffre d'affaires pour l'année en cours ?"
    },
    {
        "key": "opportunite_pays",
        "icon": "🗺️",
        "title": "Opportunités par pays",
        "subtitle": "Vue globale",
        "value": "Donne les opportunités par pays dans un tableau"
    }
]

# ---------- RENDER ----------
def render_suggestions():
    #st.markdown("### 💡 Suggestions de questions")
    st.markdown(
    "<div style='font-size:14px; font-weight:600; color:#444; margin-bottom:8px;'>"
    "💡 Suggestions de questions"
    "</div>",
    unsafe_allow_html=True
)


    cols = st.columns(3)  # 3 colonnes modernes

    for i, s in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:6px;">
                        <div style="font-size:28px;">{s['icon']}</div>
                        <div style="font-weight:600;">{s['title']}</div>
                        <div style="font-size:12px; color:gray;">
                            {s['subtitle']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "Poser la question",
                    key=f"suggestion_{s['key']}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = s["value"]
