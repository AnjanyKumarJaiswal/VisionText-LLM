from PyPDF2 import PdfReader
from PIL import Image
import fitz
import io
import torch
from transformers import AlignProcessor, AlignModel
from google import generativeai as Gemini
from dotenv import load_dotenv
import os
from huggingface_hub import login

from utils.pineconedb import PineConeVectorDB  
from models.embeddings import MultiModalEmbedder
from utils.preprocessing import Image_Text_Extractor

load_dotenv()


embedder = MultiModalEmbedder()
file_path = './utils/test.pdf'
pinecone_manager = PineConeVectorDB(embedder=embedder, pdf_path=file_path)
pinecone_manager.connect()
pinecone_manager.upsert_embedding()


Gemini.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model_name = 'models/gemini-pro'


user_query = "What is this Cat Doing?"  # Example query 

query_embedding_for_pinecone = embedder.embed_query_for_pinecone(query=user_query)

print(f"Succcesfully User Query Embedding Done")

# Retrieve relevant documents from Pinecone
retrieved_results = pinecone_manager.vector_search(query_embedding=query_embedding_for_pinecone, top_k=3)

print(f"Succcesfully Retrieved User Query :{retrieved_results}")

retrieved_context = ""
for i, result in enumerate(retrieved_results):
    text = result.metadata.get('text', "")
    retrieved_context += f"**Document {i+1}:**\n{text}\n\n"

print(f"The New Retrieved Content:{retrieved_context}")

prompt_with_context = f"""
You are a helpful AI assistant. Answer the question using the provided context. 
If the answer cannot be found in the context, say "I don't know."

Context:
{retrieved_context}

Question: {user_query}

Answer:
"""

prompt_without_context = f"""
You are a helpful AI assistant. Answer the question: {user_query}
"""

try:
    print("Starting generating with Gemini Model")
    gemini_model = Gemini.GenerativeModel(gemini_model_name)

    response_with_context = gemini_model.generate_content(prompt_with_context)
    response_without_context = gemini_model.generate_content(prompt_without_context)

    print("\n--- Response with Context ---")
    print(response_with_context.text)

    print("\n--- Response without Context ---")
    print(response_without_context.text)

except Exception as e:
    print(f"Error using Gemini Model: {e}")