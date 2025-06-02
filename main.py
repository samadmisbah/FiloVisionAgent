from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from gpt4_ranker import rank_images

app = FastAPI()

class ImageItem(BaseModel):
    name: str
    url: str

class RankRequest(BaseModel):
    images: List[ImageItem]
    history_folder: Optional[str] = None

@app.post("/rank-images")
async def rank_images_endpoint(payload: RankRequest):
    results = await rank_images(payload.images, payload.history_folder)
    return results
