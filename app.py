import streamlit as st
from utils import *

def show_website():
    st.sidebar.title("Image Captioning System")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, "Uploaded Successfully", use_column_width=True)
        image = get_image(uploaded_file)
        ok = st.sidebar.button("Generate Caption")
        if ok:
            caption = get_caption(image)
            st.warning(caption)

if __name__ == "__main__":
    show_website()