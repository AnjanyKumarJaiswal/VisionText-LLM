# Multimodal RAG Application

<!-- GitAds-Verify: EWPTK5FXPNB5A6AFQ8UKPHBXZ6LAUZ4Q -->

This project implements a multimodal Retrieval Augmented Generation (RAG) application that leverages both text and image content to provide comprehensive answers to user queries.

## Task Description

**Goal:**
Implement a multimodal query processing workflow that handles both text and image inputs, without relying on OCR for image content extraction.  Utilize a multimodal large language model (LLM) to process retrieved text and images for generating responses.

**Objectives:**

- **Build a complete pipeline:**  Create a system that extracts text and images from documents, embeds them into a shared vector space, performs semantic retrieval, and generates answers using a multimodal LLM.
- **Multimodal Embedding:**  Use a multimodal embedding model to represent both text and images in a unified vector space.
- **Semantic Retrieval:**  Utilize a vector database to efficiently retrieve relevant text and images based on user queries.
- **Multimodal LLM:**  Use a multimodal LLM to generate answers based on the retrieved text and images.
- **Avoid OCR:**  Process images directly without using OCR for image content extraction.
- **Robustness:**  Design the system to handle diverse user queries and gracefully address errors.

**Diagram:**


![Screenshot 2024-09-19 202433](https://github.com/user-attachments/assets/0637d649-8420-4cd7-9307-75b121b4985d)

## Technologies Used

- **Hugging Face Transformers:**  For loading the `AlignModel` for multimodal embedding.
- **Google Gemini:** A multimodal large language model for answer generation.
- **Pinecone:** A vector database for efficient storage and retrieval of multimodal embeddings.
- **PyPDF2:** For extracting text from PDF documents.
- **Fitz:** For extracting images from PDF documents.
- **PIL (Pillow):** For image manipulation.
- **FastAPI:** A web framework for building the API.
- **Uvicorn:** An ASGI server for running the FastAPI application.

## Code Structure

- **`models/embeddings.py`:** Contains the `MultiModalEmbedder` class for embedding text and images using the `AlignModel`.
- **`utils/pineconedb.py`:** Handles the connection to Pinecone, upserting embeddings, and performing vector search.
- **`utils/preprocessing.py`:** Extracts text and images from PDF documents.
- **`models/gemini.py`:** Implements the `GenerateGeminiResponse` class for generating answers using the Google Gemini model.
- **`test.py`:** A script for testing the multimodal RAG pipeline.
- **`main.py`:** The main FastAPI application file.

## Usage on Your Local Machine
1. **Set UP Python Venv**
   ```
   python -m venv app
   ```
2. **Change to Working Directory**
   ```
   cd src
   ```

3. **Set Up Environment:**
   - Install necessary libraries:
     ```pip install -r requirements.txt```
   - Create a ```.env``` file and add the following environment variables:
     - ```HF_KEY``` (your Hugging Face API token)
     - ```PINECONE_API_KEY``` (your Pinecone API key)
     - ```PINECONE_REGION``` (your Pinecone region)
     - ```PINECONE_INDEX``` (the name of your Pinecone index)
     - ```GEMINI_API_KEY``` (your Google Gemini API key)

4. **Run the Application:**
   - Execute to start the FastAPI application.
     ```uvicorn main:app --reload``` 
   - Access the application at `http://localhost:8000/run?query={your_query}` to get a response.
   - For example: `http://localhost:8000/run?query=What%20is%20the%20cat%20doing?`

## Notes

- The code uses a placeholder image (`Image.new('RGB', (224, 224), (255, 255, 255))`) when embedding queries, as there may not always be a relevant image available.
- You can customize the prompt engineering for the Gemini model to achieve better results.
- This project demonstrates a basic implementation of a multimodal RAG system.  You can explore various extensions and optimizations to enhance the application.
