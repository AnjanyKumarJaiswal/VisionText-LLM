from fastapi import FastAPI
import uvicorn
from utils.pineconedb import PineConeVectorDB
from models.embeddings import MultiModalEmbedder
from models.gemini import GenerateGeminiResponse
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

HF_API_KEY = os.getenv("HF_KEY")

embedder = MultiModalEmbedder(api_key=HF_API_KEY)
file_path='./../test.pdf'

pinecone_instances = PineConeVectorDB(
    api_key=os.getenv('PINECONE_API_KEY'),
    region=os.getenv('PINECONE_REGION'),
    index_name=os.getenv('PINECONE_INDEX'),
    embedder=embedder, 
    pdf_path=file_path
    )
pinecone_instances.connect()

gemini = GenerateGeminiResponse(
    Gemini_API_KEY=os.getenv('GEMINI_API_KEY'),
    pinecone=pinecone_instances,
    Embedder=embedder,
    model='models/gemini-pro'
)

pinecone_instances.upsert_embedding()

@app.get('/')
def run():
    return {"message":"This Server is Working hehe"}

@app.get('/run')
def rag_app(query:str) -> str:
    try:
        generated_respones = gemini.generate_query(query)
        return generated_respones
    except Exception as e:
        print("An Error Occured!! :(")

if __name__ == '__main__':
    uvicorn.run(app=app,host="localhost" , port=8000)