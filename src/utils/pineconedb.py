from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from models.embeddings import MultiModalEmbedder
from .preprocessing import Image_Text_Extractor

class PineConeVectorDB:
    def __init__(self, embedder: MultiModalEmbedder, pdf_path: str):
        load_dotenv()
        self.API_KEY = os.getenv('PINECONE_API_KEY')
        self.env = os.getenv('PINECONE_REGION')
        self.index_name = os.getenv('PINECONE_INDEX')
        self.index = None
        self.embedder = embedder
        self.pdf_path = pdf_path
        print(f"API_KEY:{self.API_KEY}")
        print(f"INDEX NAME:{self.index_name}")
        print(f"Region:{self.env}")

    def connect(self):
        try:
            self.pinecone = Pinecone(api_key=self.API_KEY, environment=self.env)
            print("Successfully Connected to PineCone")
            all_indexes = [index['name'] for index in self.pinecone.list_indexes()]
            if self.index_name not in all_indexes:
                print(self.index_name)
                print("Creating a new Index name in PineCone!!!")
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=1280,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=self.env)
                )
            print(f"Index Name Already exists{self.index_name}")
            self.index = self.pinecone.Index(self.index_name) 
            print("Connected to index:", self.index_name) 
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")

    def upsert_embedding(self):
        try:
            print("Starting upsert embedding")
            text_image_pairs = Image_Text_Extractor(self.pdf_path).extract_text_and_images()  
            print(f"text-embedding-pairs:{text_image_pairs}")
            text = text_image_pairs[0]
            image = text_image_pairs[1]
            embeddings = []
            for i, (text, image) in enumerate(zip(text,image)):  # Unpack the tuple
                print("Starting Embedding!!!")
                try:
                    combined_embedding = self.embedder.embed_text_and_image(text, image)
                    print("text and image embedding done")
                    embeddings.append({
                        'id': str(i),
                        'values': combined_embedding,
                        'metadata': {'text': text, 'image_index': i}
                    })
                except Exception as e:
                    print(f"Error embedding text-image pair {i}: {e}") 

            if embeddings:
                self.index.upsert(vectors=embeddings)
                print("Embeddings upserted successfully!")
            else:
                print("No embeddings were successfully created.")
        except Exception as e:
            print(f"Error upserting embeddings: {e}")

    def vector_search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise Exception("Not Connected to PineCone!")
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches