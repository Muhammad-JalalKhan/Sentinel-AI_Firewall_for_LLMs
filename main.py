import json
import numpy as np
import requests
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# --- Configuration ---
# Update this IP if your LLM is on another computer
LOCAL_LLM_URL = "http://localhost:11434/api/generate" 

# PATH SETTINGS
# We use os.path.join to make sure it works on Windows correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_JSON = os.path.join(BASE_DIR, "models", "xgb_embed_model.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

# --- Global Variables ---
models = {}

# --- Lifespan (Startup/Shutdown) Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Load Models on Startup ---
    print("--- STARTUP: Loading Models ---")
    
    # 1. Check if classification file exists
    if not os.path.exists(MODEL_PATH_JSON):
        print(f"\nCRITICAL ERROR: The file was not found!")
        print(f"Looking at location: {MODEL_PATH_JSON}")
        print("Please make sure you created a 'models' folder and put the JSON file inside.\n")
        raise FileNotFoundError(f"Missing model file at: {MODEL_PATH_JSON}")

    print("Loading Embedding Model (MiniLM)...")
    models["embedder"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading Classification Model from {MODEL_PATH_JSON}...")
    try:
        classifier = xgb.Booster()
        classifier.load_model(MODEL_PATH_JSON)
        models["classifier"] = classifier
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")
        raise e
        
    yield
    # --- Clean up on Shutdown ---
    models.clear()
    print("Models unloaded.")

app = FastAPI(lifespan=lifespan)

# --- Input Structure ---
class PromptRequest(BaseModel):
    prompt: str

# --- Inference Endpoint ---
@app.post("/inference")
async def inference_endpoint(request: PromptRequest):
    user_prompt = request.prompt
    
    # Access models from global dictionary
    embedder = models["embedder"]
    classifier = models["classifier"]
    
    # --- Step A: Vectorize ---
    embedding = embedder.encode(user_prompt)
    embedding_reshaped = np.array([embedding])
    dmatrix_input = xgb.DMatrix(embedding_reshaped)
    
    # --- Step B: Classify ---
    prediction_prob = classifier.predict(dmatrix_input)
    # Assume 1 = Malicious, 0 = Benign. Threshold 0.5
    is_malicious = prediction_prob[0] > 0.5 
    
    print(f"Prompt: {user_prompt[:30]}... | Malicious Score: {prediction_prob[0]:.4f}")

    # --- Step C: Decision Logic ---
    if is_malicious:
        return {
            "status": "refused",
            "security_check": "FAIL",
            "message": "This prompt was classified as malicious and was blocked."
        }
    else:
        # --- Step D: Forward to LLM ---
        try:
            payload = {
                "model": "llama3", # Change to llama2 if needed
                "prompt": user_prompt,
                "stream": False
            }
            response = requests.post(LOCAL_LLM_URL, json=payload)
            
            if response.status_code == 200:
                llm_response = response.json()
                return {
                    "status": "success",
                    "security_check": "PASS",
                    "llm_response": llm_response.get("response", "")
                }
            else:
                return {"status": "error", "message": "LLM Service unavailable"}
                
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to LLM: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)