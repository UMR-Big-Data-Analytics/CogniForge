import streamlit as st
from PIL import Image
from furthrmind.collection import File

from utils.furthr import FURTHRmind, download_item_bytes

st.set_page_config(page_title="CogniForge | Photo Viewer", page_icon="🖼️")

st.write("""# DB Photo Viewer

This is a simple viewer for images stored in the FURTHRmind database.""")

if 'file_id' in st.query_params:
    file = File(st.query_params.file_id)
    file.id = st.query_params.file_id
    data, filename = download_item_bytes(file)
else:
    widget = FURTHRmind()
    widget.select_file()
    data, filename = widget.download_bytes_button() or (None, None)

if data:
    st.image(data, caption=filename or st.query_params.file_id)