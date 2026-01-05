# app.py

import streamlit as st
from layout import render_header, render_footer
from sidebar import render_sidebar
from chat import (
    init_chat_state,
    render_history,
    handle_user_input,
    process_question,
    render_search
)
from suggestion import render_suggestions


def main():
    st.set_page_config(
        page_title="Neemba – GEN'AI CVM",
        page_icon="🚜",
        layout="wide"
    )

    init_chat_state()
    render_header()
    render_sidebar()

    render_history()
    render_suggestions()

    question = handle_user_input()
    if question:
        process_question(
            model="llama3.1-70b",
            temperature=0.1
        )

    render_search()
    render_footer()


if __name__ == "__main__":
    main()
