import logging

import streamlit as st

from config import Settings
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_secrets() -> dict[str, str]:
    try:
        return {key: value for key, value in st.secrets.items() if isinstance(value, str)}
    except FileNotFoundError:
        return {}


@st.cache_resource(show_spinner="Loading the models...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(Settings.load(overrides=read_secrets()))


def main() -> None:
    st.title("Multimodal RAG Application")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is None:
        st.info("Upload a PDF to start asking questions about its text and images.")
        return

    pipeline = get_pipeline()

    if st.session_state.get("file_id") != uploaded_file.file_id:
        with st.spinner("Reading and indexing the PDF..."):
            st.session_state.document = pipeline.ingest(uploaded_file.getvalue())
        st.session_state.file_id = uploaded_file.file_id
    document = st.session_state.document
    st.caption(
        f"Indexed - pages: {len(document.pages)}, "
        f"text chunks: {len(document.chunks)}, images: {len(document.images)}"
    )

    with st.form("question"):
        user_query = st.text_input("Ask me anything!")
        submitted = st.form_submit_button("Generate Answer")
    if not submitted:
        return
    if not user_query.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Generating the answer..."):
        answer = pipeline.answer(user_query, document)
    st.markdown(answer.text)

    with st.expander("Sources used for this answer"):
        if answer.pages:
            st.write("Text from page(s): " + ", ".join(str(page) for page in answer.pages))
        for item in answer.images:
            st.image(item.image, caption=f"Image from page {item.page}")


try:
    main()
except Exception as exc:
    logger.exception("The app failed")
    st.error(f"An error occurred: {exc}")
