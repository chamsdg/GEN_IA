# layout.py

# layout.py
import streamlit as st
import streamlit.components.v1 as components

HEADER_HTML = """
<style>
.hover-button {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    background-color: #1E90FF;
    color: white;
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s, background-color 0.2s;
}
.hover-button:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    background-color: #1C86EE;
    cursor: pointer;
}
</style>

<div style="
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
">
    <div style="display: flex; align-items: center; gap: 30px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTSgfw9M41EkrtiC-5aV_4x3RNVOheebqUrg&s"
             style="height:55px;">
        <div>
            <h2>GENERATIVE AI – CVM</h2>
            <p style="color:#B00020; font-size:14px; font-weight:bold; margin-top:-10px;">
                GEN'AI – CVM peut faire des erreurs. Vérifiez les réponses.
            </p>
        </div>
    </div>

    <div style="display: flex; justify-content: center; gap: 56px; margin-top: 10px;">
        <button class="hover-button">💼 Opportunités / OLGA</button>
        <button class="hover-button">🚜 Équipement / Machines</button>
        <button class="hover-button">📄 Ventes / Factures</button>
        <button class="hover-button">💶 Major Classe / Produits</button>
    </div>
</div>
"""

FOOTER_HTML = """
<hr>
<footer style="text-align:center; font-size: 0.85rem; color: #666;">
    © 2025 Neemba | Plateforme IA Générative – Développé par Marketing Intelligence
</footer>
"""

def render_header():
    components.html(HEADER_HTML, height=230, scrolling=False)

def render_footer():
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
