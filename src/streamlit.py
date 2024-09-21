import streamlit as st
from models.embeddings import MultiModalEmbedder
from models.gemini import GenerateGeminiResponse
from utils.pineconedb import PineConeVectorDB
from dotenv import load_dotenv
import os
from PIL import Image

load_dotenv()

gemini_api_key = st.secrets["GEMINI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
hf_key = st.secre["HF_KEY"]
pinecone_index = st.secrets["PINECONE_INDEX"]
pinecone_region = st.secrets-["PINECONE_REGION"]

embedder = MultiModalEmbedder(api_key=hf_key)
pinecone_instances = None  

st.title("Multimodal RAG Application")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

generate_button = st.button("Generate Answer")


if uploaded_file:
    try:
        pinecone_instances = PineConeVectorDB(
        api_key=pinecone_api_key,
        region=pinecone_region,
        index_name=pinecone_index,
        embedder=embedder,
        pdf_path=uploaded_file 
        )
        pinecone_instances.connect()
        pinecone_instances.upsert_embedding()

        gemini = GenerateGeminiResponse(
            Gemini_API_KEY=gemini_api_key,
            pinecone=pinecone_instances,
            Embedder=embedder,
            model='models/gemini-pro'
        )

        user_query = st.text_input("Ask me anything!", "")
        if generate_button:
            if user_query:
                response = gemini.generate_query(user_query)
                st.text(response)
            else:
                st.warning("Please enter a question.")
    except Exception as e:
        st.error(f"An error occurred: {e}")