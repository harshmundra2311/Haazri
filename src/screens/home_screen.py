import streamlit as st
from src.ui.base_layout import style_background_home, style_base_layout

style_base_layout()
style_background_home()
def home_screen():
    st.title("Home Screen")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Student", type="primary", width="stretch"):
            st.session_state["login_type"] = "Student"
            st.rerun()
    with col2:
        if st.button("Teacher", type="primary", width="stretch"):
            st.session_state["login_type"] = "Teacher"
            st.rerun()
