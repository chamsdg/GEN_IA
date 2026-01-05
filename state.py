import streamlit as st
def init_state():
    defaults = {
        "history": [],
        "pending_question": None,
        "suggestion_input": None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
