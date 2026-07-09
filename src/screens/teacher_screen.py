import streamlit as st

from src.ui.base_layout import style_background_dashboard, style_base_layout

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard


def teacher_screen():

    style_background_dashboard()
    style_base_layout()

    if "teacher_login_type" not in st.session_state or st.session_state.teacher_login_type == "login":
        teacher_screen_login()
    else:
        teacher_screen_register()


def teacher_screen_login():

    c1, c2 = st.columns([3, 2], vertical_alignment="center")

    with c1:
        header_dashboard()

    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="loginbackbtn",
            shortcut="control+backspace",
            width="stretch",
        ):
            st.session_state.current_page = "home"
            st.rerun()

    st.header("Login using password", text_alignment="center")

    st.space()
    st.space()

    teacher_username = st.text_input(
        "Enter username",
        placeholder="ananyaroy"
    )

    teacher_pass = st.text_input(
        "Enter password",
        type="password",
        placeholder="Enter password"
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    with btnc1:
        st.button(
            "Login",
            icon=":material/passkey:",
            shortcut="control+enter",
            width="stretch"
        )

    with btnc2:
        if st.button(
            "Register Instead",
            type="primary",
            icon=":material/passkey:",
            width="stretch"
        ):
            st.session_state.teacher_login_type = "register"
            st.rerun()

    footer_dashboard()


def teacher_screen_register():

    c1, c2 = st.columns([3, 2], vertical_alignment="center")

    with c1:
        header_dashboard()

    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="registerbackbtn",
            shortcut="control+backspace",
            width="stretch",
        ):
            st.session_state.current_page = "home"
            st.rerun()

    st.header("Register your teacher profile", text_alignment="center")

    st.space()
    st.space()

    teacher_username = st.text_input(
        "Enter username",
        placeholder="ananyaroy"
    )

    teacher_name = st.text_input(
        "Enter name",
        placeholder="Ananya Roy"
    )

    teacher_pass = st.text_input(
        "Enter password",
        type="password",
        placeholder="Enter password"
    )

    teacher_pass_confirm = st.text_input(
        "Confirm your password",
        type="password",
        placeholder="Enter password"
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    with btnc1:
        st.button(
            "Register now",
            icon=":material/passkey:",
            shortcut="control+enter",
            width="stretch"
        )

    with btnc2:
        if st.button(
            "Login Instead",
            type="primary",
            icon=":material/passkey:",
            width="stretch"
        ):
            st.session_state.teacher_login_type = "login"
            st.rerun()

    footer_dashboard()