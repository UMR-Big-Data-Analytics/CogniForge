from collections import defaultdict

import streamlit as st


def make_callback_fn(key: str):
    def click_button():
        if "clicked" not in st.session_state:
            st.session_state.clicked = defaultdict(bool)
        st.session_state.clicked[key] = True

    return click_button


def button(label: str, key: str, stateful: bool = False) -> bool:
    """Create a button that memorizes its state."""
    button = st.button(label, key=key, on_click=make_callback_fn(key))
    if stateful:
        return st.session_state.get("clicked", {}).get(key, False)

    return button
