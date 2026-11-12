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

## How It Works

1. **Extract:** the text of every page and every embedded image are pulled out of the PDF.
2. **Embed:** page text is split into small overlapping chunks, and each chunk and each image is embedded on its own with ALIGN. Both land in the same 640-dimensional space, so a text question can be compared with text and with images.
3. **Store:** the vectors go into a local Chroma database, in a collection derived from the PDF's contents. Re-uploading the same file reuses what is already indexed, and different documents never mix.
4. **Retrieve:** the question is embedded as text and the closest text chunks and the closest images are looked up separately.
5. **Generate:** a multimodal model (Gemini by default) receives the full text of the pages that matched plus the retrieved images themselves (no OCR), and writes the answer from them.

## Technologies Used

- **Hugging Face Transformers:**  For loading the `AlignModel` for multimodal embedding.
- **Google Gemini:** A multimodal large language model for answer generation, called with Google's `google-genai` SDK. An OpenAI model can be used instead.
- **ChromaDB:** A free, open-source vector database that runs locally and stores the multimodal embeddings on disk.
- **PyMuPDF:** For extracting text and images from PDF documents.
- **PIL (Pillow):** For image manipulation.
- **FastAPI:** A web framework for building the API.
- **Uvicorn:** An ASGI server for running the FastAPI application.
- **Streamlit:** For the upload-and-ask web UI.

## Code Structure

All code lives in `src/`.

- **`config.py`:** Reads the settings from environment variables, a `.env` file or Streamlit secrets.
- **`utils/preprocessing.py`:** Extracts text and images from PDF documents and splits the text into chunks.
- **`models/embeddings.py`:** Contains the `MultiModalEmbedder` class for embedding text and images using the `AlignModel`.
- **`utils/vectordb.py`:** The `ChromaVectorDB` class: storing embeddings in Chroma and performing vector search.
- **`models/llm.py`:** Implements the `GenerateGeminiResponse` class for generating answers using the Google Gemini model, and `GenerateOpenAIResponse` for doing the same with an OpenAI model.
- **`pipeline.py`:** The `RAGPipeline` class that ties the steps above together; used by both apps.
- **`main.py`:** The FastAPI application. It indexes `test.pdf` at startup.
- **`streamlit.py`:** The Streamlit application. It indexes whatever PDF you upload.

## Usage on Your Local Machine

Python 3.11 or newer is required.

1. **Set UP Python Venv**
   ```
   python -m venv app
   ```
   Activate it with `app\Scripts\activate` (Windows) or `source app/bin/activate` (macOS/Linux).

2. **Change to Working Directory**
   ```
   cd src
   ```

3. **Set Up Environment:**
   - Install necessary libraries:
     ```pip install -r requirements.txt```
   - Create a ```.env``` file (in `src/` or in the project root) and add one of the following environment variables:
     - ```GEMINI_API_KEY``` (a Gemini API key; free to create in [Google AI Studio](https://aistudio.google.com/))
     - ```OPENAI_API_KEY``` (an OpenAI API key; needs paid credits)

     If both are set, Gemini is used.
   - Optional variables:
     - ```LLM_MODEL``` (default `gemini-3.5-flash-lite` with Gemini and `gpt-6-luna` with OpenAI; any model of that provider that accepts images works, for example `gemini-3.8-flash`)
     - ```CHROMA_PATH``` (the folder for the Chroma database; default `chroma_db` in the project root)
     - ```HF_TOKEN``` (a Hugging Face token; optional, since the ALIGN model is public, but it avoids the anonymous download limits. The older name `HF_KEY` also works)
     - ```PDF_PATH``` (the PDF the FastAPI app indexes; default `test.pdf` in the project root)

4. **Run the FastAPI Application:**
   - Execute to start the FastAPI application.
     ```uvicorn main:app --reload```
   - Access the application at `http://localhost:8000/run?query={your_query}` to get a response.
   - For example: `http://localhost:8000/run?query=What%20is%20the%20cat%20doing?`
   - The response is JSON with the answer and what it was based on:
     ```json
     {"answer": "...", "pages": [1], "image_pages": [1]}
     ```

5. **Or Run the Streamlit Application** (from the project root):
   ```
   streamlit run src/streamlit.py
   ```
   Upload a PDF, then ask questions about it. Settings are read from Streamlit secrets when they exist, otherwise from the `.env` file.

## Notes

- With a Gemini key on Google's free tier, nothing here costs money: the embeddings (ALIGN) and the vector database (Chroma) run on your own machine, and no Pinecone or Chroma account is needed. The free tier limits how many questions you can ask per minute and per day (your limits are shown in Google AI Studio), and Google may use free-tier prompts to improve its products.
- The Chroma database is the `chroma_db` folder. It is safe to delete; documents are re-indexed the next time they are used.
- The first start downloads the ALIGN model (about 1.3 GB) from Hugging Face.
- ALIGN's text encoder reads at most 64 tokens, which is why text is embedded in small chunks; the answering model still receives the full page each matching chunk came from.
- You can customize the prompt engineering for the model (`SYSTEM_PROMPT` in `models/llm.py`) to achieve better results.
- This project demonstrates a basic implementation of a multimodal RAG system.  You can explore various extensions and optimizations to enhance the application.
