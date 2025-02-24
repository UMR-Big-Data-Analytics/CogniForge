import streamlit as st
from io import BytesIO, StringIO
import requests


# reuse TCP connections
session = requests.Session()


@st.cache_data
def get_as_string(
    url: str, headers: dict
) -> StringIO | requests.Response:
    """Get a string via HTTP."""
    response = session.get(url, headers=headers)
    if response.ok:
        return StringIO(response.content.decode("latin"))
    else:
        return response


@st.cache_data
def get_as_bytes(
    url: str, headers: dict
) -> BytesIO | requests.Response:
    """Get raw bytes via HTTP."""
    response = session.get(url, headers=headers)
    print(response.raw.version)
    if response.ok:
        return BytesIO(response.content)
    else:
        return response