import streamlit as st
from furthrmind.collection.baseclass import BaseClass


def selectbox(
    objects: list[BaseClass], format_name: callable = None, *args, **kwargs
) -> BaseClass | None:
    """Select an object from a list of objects."""
    if len(objects) == 0:
        return None
    if format_name is None:
        format_name = lambda o: o.name
    chosen_name = st.selectbox(options=map(format_name, objects), *args, **kwargs)
    chosen_object = next(filter(lambda o: format_name(o) == chosen_name, objects))
    return chosen_object
