from google import generativeai as Gemini
from utils.pineconedb import PineConeVectorDB
from .embeddings import MultiModalEmbedder
from dotenv import load_dotenv
import os
from PIL import Image

class GenerateGeminiResponse:
    def __init__(
        self,
        Gemini_API_KEY: str,
        pinecone: PineConeVectorDB,
        Embedder: MultiModalEmbedder,
        model: str='models/gemini-pro'
    ):
        load_dotenv()
        self.Gemini_API_KEY = Gemini_API_KEY or os.getenv('GEMINI_API_KEY')
        self.pinecone = pinecone
        self.Embedder = Embedder
        self.model = model
        Gemini.configure(api_key=self.Gemini_API_KEY)
        
    def generate_query(self, user_query:str) -> str:
        query_embedding = self.Embedder.embed_query_for_pinecone(query=user_query)
        retrieved_embedding = self.pinecone.vector_search(query_embedding=query_embedding, top_k=5)
        try:
            formatted_res =""
            for i,res in enumerate(retrieved_embedding):
                text = res.metadata.get('text',"")
                print(f"this is reteieved text: {text}")
                formatted_res+=f"**Document {i+1}:**\n{text}\n\n"
        except Exception as e:
            print(f"Error retrieving text from Pinecone: {e}")
            return "PineCone Was not Able to Fetch the Text"
        
        prompt_with_context = f"""
                    You are a helpful AI assistant. Answer the question using the provided context. 
                    If the answer cannot be found in the context, say "I don't know."

                    Context:
                    {formatted_res}

                    Question: {user_query}

                    Answer:
                    """
                    
        prompt_without_context = f"""You are a helpful AI assistant. Answer the question: {user_query}"""
        try:
            print("Starting to Generate with Gemini Model.....")
            gemini_model = Gemini.GenerativeModel(self.model)
            response_with_context = gemini_model.generate_content(prompt_with_context)
            response_without_context = gemini_model.generate_content(prompt_without_context)
            print("\n--- Response with Context ---")
            print(response_with_context.text)
            print("\n--- Response without Context ---")
            print(response_without_context.text)
            return response_with_context.text 
        except Exception as e:
            print(f"Error using Gemini Model: {e}")  
            return "An error occurred while generating a response. Please try again later."
        