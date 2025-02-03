import streamlit as st
from io import BytesIO, StringIO
import requests


@st.cache_data
def get_as_string(
    url: str, headers: dict, _chunk_size: int = 8192000
) -> StringIO | requests.Response:
    """Get a string via HTTP."""
    response = requests.get(url, headers=headers)
    if response.ok:
        bytes = b""
        for chunk in response.iter_content(chunk_size=_chunk_size):
            if chunk:
                bytes += chunk
        return StringIO(bytes.decode("latin"))
    else:
        return response


@st.cache_data
def get_as_bytes(
    url: str, headers: dict
) -> BytesIO | requests.Response:
    """Get raw bytes via HTTP."""
    response = requests.get(url, headers=headers)
    if response.ok:
        return BytesIO(response.content)
    else:
        return response