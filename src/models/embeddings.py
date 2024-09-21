from huggingface_hub import login
from dotenv import load_dotenv
import os
from PIL import Image
import torch
from transformers import AlignProcessor, AlignModel

load_dotenv()

hf_key = os.getenv('HF_KEY')
login(hf_key)

class MultiModalEmbedder:
    def __init__(self, align_model_name='kakaobrain/align-base'):
        self.processor = AlignProcessor.from_pretrained(align_model_name)
        self.model = AlignModel.from_pretrained(align_model_name)

    def embed_text_and_image(self, text: str, image: Image.Image) -> list[float]:
        print(f"Embedding text: '{text}'")
        print(f"Image shape: {image.size}")

        inputs = self.processor(text=text, images=image, return_tensors='pt')
        inputs['pixel_values'] = inputs.pixel_values

        with torch.no_grad():
            outputs = self.model(**inputs)

        combined_embedding = outputs.text_embeds.squeeze().tolist() + outputs.image_embeds.squeeze().tolist()
        print("\n\nSuccessfully extracted text and image embeddings\n\n")
        print(f"Combined embedding shape: {len(combined_embedding)}")
        return combined_embedding

    def embed_query_for_pinecone(self, query: str) -> list[float]:
        # print(f"Embedding query: '{query}' for Pinecone search...")
        placeholder_image = Image.new('RGB', (224, 224), (255, 255, 255))
        return self.embed_text_and_image(text=query, image=placeholder_image)