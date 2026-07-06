from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import uuid
import json
import os
import re
import logging
import signal
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import aws_service
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

PORT = int(os.getenv('PORT', '5000'))
MODEL_PATH = os.getenv('MODEL_PATH', 'volcano_classifier_model.joblib')
ENCODER_PATH = os.getenv('ENCODER_PATH', 'volcano_label_encoder.joblib')

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
    input: Dict[str, Any]

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
    mode: str
    aws_enabled: bool
    port: int

class TrainingDataInput(BaseModel):
    """Manual training data input"""
    tinggi_meter: float = Field(..., ge=0, le=10000, description="Height in meters")
    lat: float = Field(..., ge=-90, le=90, description="Longitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    bentuk: str = Field(..., description="Actual shape class of the volcano")

class VerifyLogInput(BaseModel):
    """Verify past prediction log input"""
    id: str = Field(..., description="Inference log UUID")
    bentuk: str = Field(..., description="Correct actual volcano shape class")

# ============================================
# GLOBAL VARIABLES
# ============================================

model = None
label_encoder = None
feature_columns = ['tinggi_meter', 'lat', 'lon']
is_ready = False
should_exit = False

# ============================================
# SIGNAL HANDLERS
# ============================================

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global should_exit
    logger.info(f"Received signal {sig}")
    should_exit = True
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_or_train_model():
    """Load model from file or train new one"""
    global model, label_encoder, is_ready
    
    logger.info("=" * 50)
    logger.info("LOADING/TRAINING MODEL...")
    logger.info("=" * 50)
    
    model_loaded = False
    
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"[OK] Model loaded from {MODEL_PATH}")
            model_loaded = True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
    
    # Load label encoder
    if os.path.exists(ENCODER_PATH):
        try:
            label_encoder = joblib.load(ENCODER_PATH)
            logger.info(f"[OK] Label encoder loaded from {ENCODER_PATH}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load label encoder: {e}")
            label_encoder = None
    
    # If model loaded but encoder is missing, create a default encoder
    if model_loaded and label_encoder is None:
        logger.warning("[WARN] Model loaded but label encoder is missing, creating default encoder")
        label_encoder = create_default_label_encoder()
    
    # If model not loaded, train new one
    if not model_loaded:
        logger.info("Training new model...")
        try:
            model, label_encoder = train_default_model()
            logger.info("[OK] New model trained and saved")
        except Exception as e:
            logger.error(f"[ERROR] Failed to train model: {e}")
            model = create_dummy_model()
            label_encoder = create_default_label_encoder()
            logger.warning("[WARN] Using dummy model as fallback")
    
    is_ready = model is not None
    logger.info("=" * 50)
    logger.info(f"Model ready: {is_ready}")
    if is_ready and label_encoder:
        logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info("=" * 50)
    
    return model, label_encoder

def train_default_model():
    """Train a default model from public dataset"""
    try:
        logger.info("Downloading training data...")
        url = 'https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv'
        df = pd.read_csv(url)
        logger.info(f"[OK] Downloaded {len(df)} records")
        
        df['tinggi_meter'] = df['tinggi_meter'].astype(str).str.extract('(\d+)').astype(float)
        
        def extract_coordinates(geolokasi_str):
            if pd.isna(geolokasi_str):
                return np.nan, np.nan
            match = re.search(r'([-+]?\d+\.?\d*)\xb0([NS])\s+([-+]?\d+\.?\d*)\xb0([EW])',
                              str(geolokasi_str), re.IGNORECASE)
            if match:
                lat_val = float(match.group(1))
                lat_dir = match.group(2).upper()
                lon_val = float(match.group(3))
                lon_dir = match.group(4).upper()
                latitude = lat_val if lat_dir == 'N' else -lat_val
                longitude = lon_val if lon_dir == 'E' else -lon_val
                return latitude, longitude
            return np.nan, np.nan

        df[['lat', 'lon']] = df['geolokasi'].apply(lambda x: pd.Series(extract_coordinates(x)))
        df_cleaned = df.dropna().copy()
        logger.info(f"[OK] Cleaned {len(df_cleaned)} records")
        
        df_final = df_cleaned[['tinggi_meter', 'lat', 'lon', 'bentuk']].copy()
        
        X = df_final[['tinggi_meter', 'lat', 'lon']]
        y = df_final['bentuk']
        
        new_label_encoder = LabelEncoder()
        y_encoded = new_label_encoder.fit_transform(y)
        
        new_model = RandomForestClassifier(random_state=42, n_estimators=100)
        new_model.fit(X, y_encoded)
        
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_label_encoder, ENCODER_PATH)
        
        logger.info(f"[OK] Model trained with {len(df_final)} samples")
        logger.info(f"Classes: {list(new_label_encoder.classes_)}")
        
        return new_model, new_label_encoder
    except Exception as e:
        logger.error(f"[ERROR] Failed to train model: {e}")
        raise

def create_dummy_model():
    """Create a dummy model as fallback"""
    logger.warning("[WARN] Creating dummy model as fallback")
    dummy_model = RandomForestClassifier(random_state=42)
    X_dummy = np.array([[1000, 0, 0], [2000, 0, 0], [3000, 0, 0]])
    y_dummy = np.array([0, 1, 2])
    dummy_model.fit(X_dummy, y_dummy)
    return dummy_model

def create_default_label_encoder():
    """Create default label encoder with known classes"""
    classes = [
        'Fumarol', 'bawah laut', 'kaldera', 'kerucut bara', 
        'kompleks', 'kubah lava', 'perisai', 'stratovulkan', 'supervulkan'
    ]
    le = LabelEncoder()
    le.fit(classes)
    return le

def predict_single(height: float, lat: float, lon: float):
    """
    Make prediction for single volcano
    
    Returns:
        tuple: (prediction_class, confidence_score)
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    input_data = pd.DataFrame([[height, lat, lon]], columns=feature_columns)
    prediction_encoded = model.predict(input_data)[0]
    
    try:
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))
    except:
        confidence = 0.5
    
    return prediction_encoded, confidence

def get_class_name(encoded_class: int) -> str:
    """
    Map encoded class to actual volcano shape name
    """
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([encoded_class])[0]
        except Exception:
            pass

    classes = [
        'Fumarol', 'bawah laut', 'kaldera', 'kerucut bara', 
        'kompleks', 'kubah lava', 'perisai', 'stratovulkan', 'supervulkan'
    ]
    
    if 0 <= encoded_class < len(classes):
        return classes[encoded_class]
    return f"unknown_class_{encoded_class}"

# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, label_encoder, is_ready
    
    logger.info("=" * 50)
    logger.info("STARTING VOLCANO CLASSIFIER API")
    logger.info("=" * 50)
    logger.info(f"Port: {PORT}")
    logger.info(f"Working Directory: {os.getcwd()}")
    
    try:
        aws_service.init_resources()
        logger.info(f"Mode: {'AWS' if aws_service.AWS_ENABLED else 'LOCAL'}")
        if not aws_service.AWS_ENABLED:
            logger.info("   AWS not configured - using local file storage")
    except Exception as e:
        logger.warning(f"[WARN] AWS initialization warning: {e}")
        logger.info("Mode: LOCAL (AWS disabled)")
    
    logger.info("=" * 50)
    
    try:
        load_or_train_model()
        is_ready = model is not None
    except Exception as e:
        logger.error(f"[ERROR] Critical error loading model: {e}")
        is_ready = False
    
    logger.info("=" * 50)
    if is_ready:
        logger.info(f"[OK] API READY on port {PORT}")
        logger.info(f"Documentation: http://localhost:{PORT}/docs")
        logger.info(f"Classes: {list(label_encoder.classes_) if label_encoder else 'Unknown'}")
    else:
        logger.error("[ERROR] API STARTED WITH ERRORS - Model not available")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Shutting down...")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Volcano Shape Classifier API",
        "version": "1.0.0",
        "mode": "AWS" if aws_service.AWS_ENABLED else "Local",
        "status": "ready" if is_ready else "loading",
        "port": PORT,
        "endpoints": "/predict, /predict/batch, /health, /info, /logs, /add-training-data, /verify-prediction, /retrain"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - simple and fast"""
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
        model_file=MODEL_PATH,
        mode="AWS" if aws_service.AWS_ENABLED else "Local",
        aws_enabled=aws_service.AWS_ENABLED,
        port=PORT
    )

@app.get("/ready")
async def ready():
    """Kubernetes readiness probe"""
    return {
        "ready": is_ready,
        "model_loaded": is_ready,
        "port": PORT
    }

@app.get("/live")
async def live():
    """Kubernetes liveness probe"""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "port": PORT
    }

@app.get("/info")
async def get_info():
    """Get model information"""
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    return {
        "success": True,
        "model_type": type(model).__name__ if model else None,
        "features": feature_columns,
        "n_features": len(feature_columns),
        "is_trained": hasattr(model, 'predict') if model else False,
        "has_probability": hasattr(model, 'predict_proba') if model else False,
        "mode": "AWS" if aws_service.AWS_ENABLED else "Local",
        "classes": list(label_encoder.classes_) if label_encoder else [],
        "port": PORT
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_volcano(volcano: VolcanoInput):
    """
    Predict volcano shape for a single volcano
    """
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Please try again later."
        )
    
    try:
        prediction_encoded, confidence = predict_single(
            volcano.tinggi_meter,
            volcano.lat,
            volcano.lon
        )
        
        prediction_class = get_class_name(prediction_encoded)
        pred_id = str(uuid.uuid4())
        
        try:
            aws_service.log_inference_to_dynamodb(
                pred_id,
                volcano.tinggi_meter,
                volcano.lat,
                volcano.lon,
                prediction_class,
                confidence
            )
        except Exception as e:
            logger.warning(f"[WARN] Failed to log inference: {e}")
        
        sqs_payload = {
            "id": pred_id,
            "tinggi_meter": float(volcano.tinggi_meter),
            "lat": float(volcano.lat),
            "lon": float(volcano.lon),
            "prediction": prediction_class,
            "confidence": float(confidence),
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            aws_service.push_to_queue(json.dumps(sqs_payload))
        except Exception as e:
            logger.warning(f"[WARN] Failed to push to queue: {e}")
        
        try:
            aws_service.log_metric("PredictionConfidence", confidence, "Percent")
        except Exception as e:
            logger.warning(f"[WARN] Failed to log metric: {e}")
        
        if confidence < 0.5:
            warning_msg = f"Low confidence volcano prediction warning!\nID: {pred_id}\nCoordinates: {volcano.lat}, {volcano.lon}\nHeight: {volcano.tinggi_meter}m\nPredicted: {prediction_class}\nConfidence: {round(confidence * 100, 2)}%"
            try:
                aws_service.send_alert(warning_msg)
            except Exception as e:
                logger.warning(f"[WARN] Failed to send alert: {e}")
        
        return PredictionResponse(
            success=True,
            prediction=prediction_class,
            confidence=round(confidence, 4),
            input={
                "id": pred_id,
                "tinggi_meter": volcano.tinggi_meter,
                "lat": volcano.lat,
                "lon": volcano.lon
            }
        )
    
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: VolcanoBatchInput):
    """
    Predict volcano shapes for multiple volcanoes
    """
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
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
            pred_id = str(uuid.uuid4())
            
            try:
                aws_service.log_inference_to_dynamodb(
                    pred_id,
                    volcano.tinggi_meter,
                    volcano.lat,
                    volcano.lon,
                    prediction_class,
                    confidence
                )
            except:
                pass
            
            results.append({
                "input": {
                    "id": pred_id,
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
    """
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    try:
        prediction_encoded, confidence = predict_single(tinggi_meter, lat, lon)
        prediction_class = get_class_name(prediction_encoded)
        pred_id = str(uuid.uuid4())
        
        return {
            "success": True,
            "prediction": prediction_class,
            "confidence": round(confidence, 4),
            "input": {
                "id": pred_id,
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

# ============================================
# ADDITIONAL ENDPOINTS
# ============================================

@app.post("/add-training-data")
async def add_training_data(data: TrainingDataInput):
    """Add verified custom training data"""
    try:
        id_str = aws_service.add_labeled_sample_to_dynamodb(
            data.tinggi_meter,
            data.lat,
            data.lon,
            data.bentuk
        )
        if id_str:
            return {
                "success": True,
                "message": "Labeled training sample added successfully.",
                "id": id_str
            }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save training sample."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/verify-prediction")
async def verify_prediction(data: VerifyLogInput):
    """Verify/correct a past prediction log"""
    try:
        success = aws_service.verify_prediction_log(data.id, data.bentuk)
        if success:
            return {
                "success": True,
                "message": f"Log '{data.id}' verified and promoted to training sample successfully."
            }
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction log with ID '{data.id}' not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/logs")
async def get_logs():
    """Retrieve all prediction logs and verified training samples"""
    try:
        logs = aws_service.get_all_logs_from_dynamodb()
        logs_sorted = sorted(
            logs, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        return {
            "success": True,
            "total": len(logs_sorted),
            "logs": logs_sorted
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with latest data"""
    global model, label_encoder, is_ready
    try:
        logger.info("Retraining pipeline initiated...")
        
        url = 'https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv'
        df = pd.read_csv(url)
        
        df['tinggi_meter'] = df['tinggi_meter'].astype(str).str.extract('(\d+)').astype(float)
        
        def extract_coordinates(geolokasi_str):
            if pd.isna(geolokasi_str):
                return np.nan, np.nan
            match = re.search(r'([-+]?\d+\.?\d*)\xb0([NS])\s+([-+]?\d+\.?\d*)\xb0([EW])',
                              str(geolokasi_str), re.IGNORECASE)
            if match:
                lat_val = float(match.group(1))
                lat_dir = match.group(2).upper()
                lon_val = float(match.group(3))
                lon_dir = match.group(4).upper()
                latitude = lat_val if lat_dir == 'N' else -lat_val
                longitude = lon_val if lon_dir == 'E' else -lon_val
                return latitude, longitude
            return np.nan, np.nan

        df[['lat', 'lon']] = df['geolokasi'].apply(lambda x: pd.Series(extract_coordinates(x)))
        df_cleaned = df.dropna().copy()
        df_final = df_cleaned[['tinggi_meter', 'lat', 'lon', 'bentuk']].copy()
        
        try:
            db_items = aws_service.get_training_samples_from_dynamodb()
            logger.info(f"Found {len(db_items)} verified custom training samples.")
        except:
            db_items = []
        
        if db_items:
            custom_records = []
            for item in db_items:
                custom_records.append({
                    'tinggi_meter': float(item['tinggi_meter']),
                    'lat': float(item['lat']),
                    'lon': float(item['lon']),
                    'bentuk': str(item['bentuk'])
                })
            df_custom = pd.DataFrame(custom_records)
            df_final = pd.concat([df_final, df_custom], ignore_index=True)
        
        X = df_final[['tinggi_meter', 'lat', 'lon']]
        y = df_final['bentuk']
        
        new_label_encoder = LabelEncoder()
        y_encoded = new_label_encoder.fit_transform(y)
        
        new_model = RandomForestClassifier(random_state=42, n_estimators=100)
        new_model.fit(X, y_encoded)
        
        score = float(new_model.score(X, y_encoded))
        
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_label_encoder, ENCODER_PATH)
        
        try:
            aws_service.upload_model_to_s3(MODEL_PATH, MODEL_PATH)
            aws_service.upload_model_to_s3(ENCODER_PATH, ENCODER_PATH)
        except:
            pass
        
        model = new_model
        label_encoder = new_label_encoder
        is_ready = True
        
        logger.info("Model successfully retrained and hot-reloaded!")
        
        return {
            "success": True,
            "message": "Model retrained and hot-reloaded successfully.",
            "metrics": {
                "training_score": round(score, 4),
                "total_samples": len(df_final),
                "custom_samples_used": len(db_items),
                "classes": list(label_encoder.classes_)
            }
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 5000))
    
    print("=" * 60)
    print(f"Starting Volcano Classifier API")
    print(f"Port: {port}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 60)
    
    # Run with proper settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
        loop="asyncio"
    )