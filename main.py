from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]

# ============================================
# DATA MODELS (Pydantic)
# ============================================

class VolcanoInput(BaseModel):
    """Single volcano input data"""
    tinggi_meter: float = Field(..., ge=0, le=10000, description="Height in meters")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    
    @validator('tinggi_meter')
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError('Height must be greater than 0')
        return v

class VolcanoBatchInput(BaseModel):
    """Batch volcano input data"""
    data: List[VolcanoInput] = Field(..., description="List of volcano data")

class PredictionResponse(BaseModel):
    """Single prediction response"""
    success: bool
    prediction: str
    confidence: float
    input: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    success: bool
    results: List[Dict[str, Any]]
    total: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_file: str

# ============================================
# GLOBAL VARIABLES
# ============================================

model = None
label_encoder = None
feature_columns = ['tinggi_meter', 'lat', 'lon']

# ============================================
# LIFESPAN MANAGEMENT (Startup & Shutdown)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    print("=" * 50)
    print("LOADING MODEL...")
    print("=" * 50)
    
    try:
        model = joblib.load('volcano_classifier_model.joblib')
        print(model)
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down...")

# ============================================
# INITIALIZE FASTAPI APP
# ============================================

app = FastAPI(
    title="Volcano Shape Classifier API",
    description="API for predicting volcano shapes based on height and coordinates",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# HELPER FUNCTIONS
# ============================================

def predict_single(height: float, lat: float, lon: float):
    """
    Make prediction for single volcano
    
    Returns:
        tuple: (prediction_class, confidence_score)
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Create DataFrame with correct feature order
    input_data = pd.DataFrame([[height, lat, lon]], columns=feature_columns)
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    
    # Get confidence (probability)
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(np.max(probabilities))
    
    # Note: Since we don't have label_encoder from notebook,
    # we need to extract classes from model or define manually
    # For now, we'll return the encoded value and you can map it later
    # Alternatively, train and save label_encoder separately
    
    return prediction_encoded, confidence

def get_class_name(encoded_class: int) -> str:
    """
    Map encoded class to actual volcano shape name
    These classes are from the notebook's label_encoder.classes_
    """
    # These are the classes from the original notebook
    classes = [
        'Fumarol', 'bawah laut', 'kaldera', 'kerucut bara', 
        'kompleks', 'kubah lava', 'perisai', 'stratovulkan', 'supervulkan'
    ]
    
    if 0 <= encoded_class < len(classes):
        return classes[encoded_class]
    return f"unknown_class_{encoded_class}"

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Volcano Shape Classifier API",
        "version": "1.0.0",
        "endpoints": "/predict, /predict/batch, /health, /info"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_file="volcano_classifier_model.joblib"
    )

@app.get("/info")
async def get_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "success": True,
        "model_type": type(model).__name__,
        "features": feature_columns,
        "n_features": len(feature_columns),
        "is_trained": hasattr(model, 'predict'),
        "has_probability": hasattr(model, 'predict_proba')
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_volcano(volcano: VolcanoInput):
    """
    Predict volcano shape for a single volcano
    
    Example request:
    {
        "tinggi_meter": 1500,
        "lat": -7.0,
        "lon": 110.0
    }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model file."
        )
    
    try:
        # Make prediction
        prediction_encoded, confidence = predict_single(
            volcano.tinggi_meter,
            volcano.lat,
            volcano.lon
        )
        
        # Get class name
        prediction_class = get_class_name(prediction_encoded)
        
        return PredictionResponse(
            success=True,
            prediction=prediction_class,
            confidence=round(confidence, 4),
            input={
                "tinggi_meter": volcano.tinggi_meter,
                "lat": volcano.lat,
                "lon": volcano.lon
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: VolcanoBatchInput):
    """
    Predict volcano shapes for multiple volcanoes
    
    Example request:
    {
        "data": [
            {"tinggi_meter": 1500, "lat": -7.0, "lon": 110.0},
            {"tinggi_meter": 2801, "lat": 4.914, "lon": 96.329},
            {"tinggi_meter": 617, "lat": 5.820, "lon": 95.280}
        ]
    }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    results = []
    
    for volcano in batch.data:
        try:
            prediction_encoded, confidence = predict_single(
                volcano.tinggi_meter,
                volcano.lat,
                volcano.lon
            )
            
            prediction_class = get_class_name(prediction_encoded)
            
            results.append({
                "input": {
                    "tinggi_meter": volcano.tinggi_meter,
                    "lat": volcano.lat,
                    "lon": volcano.lon
                },
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "input": {
                    "tinggi_meter": volcano.tinggi_meter,
                    "lat": volcano.lat,
                    "lon": volcano.lon
                },
                "error": str(e),
                "status": "failed"
            })
    
    return BatchPredictionResponse(
        success=True,
        results=results,
        total=len(results)
    )

@app.post("/predict/form")
async def predict_form(
    tinggi_meter: float,
    lat: float,
    lon: float
):
    """
    Predict using form data (for web forms)
    
    Example: POST with form-urlencoded
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        prediction_encoded, confidence = predict_single(tinggi_meter, lat, lon)
        prediction_class = get_class_name(prediction_encoded)
        
        return {
            "success": True,
            "prediction": prediction_class,
            "confidence": round(confidence, 4),
            "input": {
                "tinggi_meter": tinggi_meter,
                "lat": lat,
                "lon": lon
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )