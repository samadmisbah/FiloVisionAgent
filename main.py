from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from gpt4_ranker import rank_images

app = FastAPI()

class ImageItem(BaseModel):
    name: str
    url: str
    id: Optional[str] = None

class RankRequest(BaseModel):
    images: List[ImageItem]
    history_folder: Optional[str] = None
    water_well_name: Optional[str] = None
    max_selections: Optional[int] = 10

@app.post("/rank-images")
async def rank_images_endpoint(payload: RankRequest):
    # Convert Pydantic models to dictionaries
    images_dict = [img.dict() for img in payload.images]
    
    results = await rank_images(
        images_dict,  # Pass as dictionaries, not Pydantic objects
        payload.history_folder,
        payload.water_well_name,
        payload.max_selections
    )
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "filo-vision-agent"}

@app.get("/")
async def root():
    return {"message": "FiloVisionAgent is live and ready."}
