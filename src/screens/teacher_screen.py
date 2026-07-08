import streamlit as st

def teacher_screen():
    st.title("Teacher Screen")
    if st.button("Back"):
        st.session_state["login_type"] = None
        st.rerun()