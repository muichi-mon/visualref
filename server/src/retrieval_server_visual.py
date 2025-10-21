
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.services.retrieval_service import RetrievalServiceVisual
from src.utils.image_utils import image_to_base64
from src.utils.utils import load_yaml

app = FastAPI(title="Retrieval Server")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


class ProcessApplyFeedbackRequest(BaseModel):
    query: str
    top_k: int
    relevant_image_paths: List[str]
    relevant_captions: str
    irrelevant_captions: str
    annotator_json_boxes_list: List[Any]
    fuse_initial_query: bool = False


class ProcessApplyFeedbackResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


retrieval_service: Optional[RetrievalServiceVisual] = None


@app.on_event("startup")
async def startup_event():
    global retrieval_service

    config = load_yaml(settings.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrieval_service = RetrievalServiceVisual(
        config=config,
        faiss_index=settings.index_path,
        device=device,
    )


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    try:
        images, scores, image_paths = retrieval_service.search_images(request.query, request.top_k)
        images = [image_to_base64(img) for img in images]
        return SearchResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Search completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_feedback", response_model=ProcessApplyFeedbackResponse)
async def apply_feedback(request: ProcessApplyFeedbackRequest):
    try:
        images, scores, image_paths = retrieval_service.process_and_apply_feedback(
            query=request.query,
            top_k=request.top_k,
            relevant_image_paths=request.relevant_image_paths,
            relevant_captions=request.relevant_captions,
            irrelevant_captions=request.irrelevant_captions,
            annotator_json_boxes_list=request.annotator_json_boxes_list,
            fuse_initial_query=request.fuse_initial_query
        )
        images = [image_to_base64(img) for img in images]
        return ProcessApplyFeedbackResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Feedback applied successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
