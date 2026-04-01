"""
api.py
FastAPI wrapper around the CNN + RAG pipeline.
Usage: uvicorn api:app --reload
Curl example for testing:
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from predict import load_model, predict
from rag import RAGPipeline

app = FastAPI(title="Deepfake Detection API")

# Lazy-loaded globals — loaded on first request, not at startup
_model = None
_rag = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def get_rag():
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
        _rag.load()
    return _rag

# health check endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "Deepfake Detection API is running"}

# main predict endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Upload an image, get back label, confidence, and RAG explanation."""
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only jpg/jpeg/png allowed.")

    temporary_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    with open(temporary_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result      = predict(temporary_path, get_model())
        label       = result['label']
        confidence  = result['confidence'] / 100
        explanation = get_rag().explain(label.upper(), confidence)

        return JSONResponse({
            "label":       label,
            "confidence":  result['confidence'],
            "explanation": explanation
        })
    finally:
        os.remove(temporary_path)
        
# """
# api.py
# FastAPI wrapper around the CNN + RAG pipeline.
# Usage: uvicorn api:app --reload
# Curl example for testing:
# curl -X POST "http://localhost:8000/predict -F "file=@path/to/image.jpg"
# """

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import shutil
# import os
# import uuid # generates unique IDs for temp files so that parallel requests don't conflict/clash

# from predict import load_model, predict
# from rag import RAGPipeline

# app = FastAPI(title = "Deepfake Detection API")

# # Loading the model and RAG once at startup - not on every request
# model = load_model()
# rag = RAGPipeline()
# rag.load()

# # health check endpoint
# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Deepfake Detection API is running"}

# # main predict endpoint
# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     """Upload an image, get back label, confidence, and RAG explanation."""
#     # validate file type
#     if not file.filename.endswith((".jpg", ".jpeg", ".png")):
#         raise HTTPException(status_code=400, detail="Invalid file type. Only jpg/jpeg/png allowed.")

#     # save uploaded file to a temp location
#     temporary_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"

#     with open(temporary_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer) # streams the file to disk in chunks, instead of loading whole file into memory at once

#     try:
#         # Step 1: Xcepttion classifies the image
#         result   =  predict(temporary_path, model)
#         label    =  result['label']
#         confidence = result['confidence'] / 100 # 0-1 for RAG prompt formatting

#         # Step 2: RAG explains the prediction using research papers
#         explanation = rag.explain(label.upper(), confidence) # upper for 'fake'/'real' to match prompt formatting 'FAKE'/'REAL'

#         return JSONResponse({
#             "label" :       label,
#             "confidence":   result['confidence'],
#             "explanation":   explanation
#         })
    
#     finally:
#         os.remove(temporary_path) # always runs, even if something above crashes. Ensures we don't fill up disk with temp files.