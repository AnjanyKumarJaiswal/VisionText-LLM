import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request

from config import Settings
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    app.state.pipeline = RAGPipeline(settings)
    app.state.document = app.state.pipeline.ingest(settings.pdf_path)
    yield


app = FastAPI(title="Multimodal RAG Application", lifespan=lifespan)


@app.get('/')
def run():
    return {"message": "This Server is Working hehe"}


@app.get('/run')
def rag_app(request: Request, query: str = Query(min_length=1)) -> dict:
    state = request.app.state
    try:
        answer = state.pipeline.answer(query, state.document)
    except Exception as exc:
        logger.exception("Could not answer query %r", query)
        raise HTTPException(status_code=502, detail=f"Could not generate an answer: {exc}") from exc
    return {
        "answer": answer.text,
        "pages": answer.pages,
        "image_pages": [item.page for item in answer.images],
    }


if __name__ == '__main__':
    uvicorn.run(app=app, host="localhost", port=8000)
