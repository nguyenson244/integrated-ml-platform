from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.recommender import HybridRecommender
import os

app = FastAPI(title="E-Commerce Hybrid Recommendation API", version="1.0")

# Load model globally on startup
model_path = os.path.join("models_saved", "hybrid_recommender.pkl")
recommender = None

@app.on_event("startup")
def load_model():
    global recommender
    if os.path.exists(model_path):
        recommender = HybridRecommender.load_model(model_path)
        print("[SUCCESS] Recommendation Model loaded.")
    else:
        print("[WARNING] Model not found. Please run the training pipeline first.")

class RecommendRequest(BaseModel):
    product_id: int
    customer_region: str
    top_n: int = 5

@app.get("/")
def read_root():
    return {"message": "Welcome to the E-Commerce Hybrid Recommendation API!"}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the pipeline first.")
    
    # Get recommendations
    recs = recommender.recommend(req.product_id, req.customer_region, top_n=req.top_n)
    
    if not recs:
        raise HTTPException(status_code=404, detail="Product ID not found or no recommendations available.")
        
    return {"product_id": req.product_id, "region": req.customer_region, "recommendations": recs}
