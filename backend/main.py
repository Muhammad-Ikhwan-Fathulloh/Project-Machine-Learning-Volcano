from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import joblib
import pandas as pd
import numpy as np
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import aws_service

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

class TrainingDataInput(BaseModel):
    """Manual training data input"""
    tinggi_meter: float = Field(..., ge=0, le=10000, description="Height in meters")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
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

# ============================================
# LIFESPAN MANAGEMENT (Startup & Shutdown)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize S3/DynamoDB resources and load model
    global model, label_encoder
    print("=" * 50)
    print("INITIALIZING AWS RESOURCES...")
    print("=" * 50)
    
    # Check/Create DynamoDB Table, S3 Bucket, SNS Topic, SQS Queue
    aws_service.init_resources()
    
    print("=" * 50)
    print("LOADING MODEL & LABEL ENCODER...")
    print("=" * 50)
    
    # Try downloading the latest model from S3
    aws_service.download_model_from_s3('volcano_classifier_model.joblib', 'volcano_classifier_model.joblib')
    aws_service.download_model_from_s3('volcano_label_encoder.joblib', 'volcano_label_encoder.joblib')
    
    try:
        model = joblib.load('volcano_classifier_model.joblib')
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        
    try:
        label_encoder = joblib.load('volcano_label_encoder.joblib')
        print("✅ Label Encoder loaded successfully!")
        print(f"🏷️ Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"ℹ️ Label Encoder not loaded (will use fallback mapping): {e}")
        label_encoder = None
        
    print("=" * 50)
    
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
    Uses the label_encoder dynamically, falling back to hardcoded classes if unavailable.
    """
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([encoded_class])[0]
        except Exception:
            pass

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
    Predict volcano shape for a single volcano and log to S3/DynamoDB/SQS/SNS/CloudWatch
    
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
        
        # Generate unique prediction ID
        pred_id = str(uuid.uuid4())
        
        # 1. Log to DynamoDB Volcano_Dataset table (is_training_sample = False)
        aws_service.log_inference_to_dynamodb(
            pred_id,
            volcano.tinggi_meter,
            volcano.lat,
            volcano.lon,
            prediction_class,
            confidence
        )
        
        # 2. Push log payload to SQS Queue
        sqs_payload = {
            "id": pred_id,
            "tinggi_meter": float(volcano.tinggi_meter),
            "lat": float(volcano.lat),
            "lon": float(volcano.lon),
            "prediction": prediction_class,
            "confidence": float(confidence),
            "timestamp": datetime.utcnow().isoformat()
        }
        aws_service.push_to_queue(json.dumps(sqs_payload))
        
        # 3. Log metric to CloudWatch
        aws_service.log_metric("PredictionConfidence", confidence)
        
        # 4. Trigger alert warning via SNS (and Telegram if configured) on low confidence
        if confidence < 0.5:
            warning_msg = f"Low confidence volcano prediction warning!\nID: {pred_id}\nCoordinates: {volcano.lat}, {volcano.lon}\nHeight: {volcano.tinggi_meter}m\nPredicted: {prediction_class}\nConfidence: {round(confidence * 100, 2)}%"
            aws_service.send_alert(warning_msg)
        
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: VolcanoBatchInput):
    """
    Predict volcano shapes for multiple volcanoes and log each to DynamoDB/SQS
    
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
            pred_id = str(uuid.uuid4())
            
            # Log individual inference to DynamoDB and SQS
            aws_service.log_inference_to_dynamodb(
                pred_id,
                volcano.tinggi_meter,
                volcano.lat,
                volcano.lon,
                prediction_class,
                confidence
            )
            
            sqs_payload = {
                "id": pred_id,
                "tinggi_meter": float(volcano.tinggi_meter),
                "lat": float(volcano.lat),
                "lon": float(volcano.lon),
                "prediction": prediction_class,
                "confidence": float(confidence),
                "timestamp": datetime.utcnow().isoformat()
            }
            aws_service.push_to_queue(json.dumps(sqs_payload))
            aws_service.log_metric("PredictionConfidence", confidence)
            
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
    Predict using form data (for web forms) and log to S3/DynamoDB
    
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
        pred_id = str(uuid.uuid4())
        
        # Log to DynamoDB and SQS
        aws_service.log_inference_to_dynamodb(
            pred_id,
            tinggi_meter,
            lat,
            lon,
            prediction_class,
            confidence
        )
        
        sqs_payload = {
            "id": pred_id,
            "tinggi_meter": float(tinggi_meter),
            "lat": float(lat),
            "lon": float(lon),
            "prediction": prediction_class,
            "confidence": float(confidence),
            "timestamp": datetime.utcnow().isoformat()
        }
        aws_service.push_to_queue(json.dumps(sqs_payload))
        aws_service.log_metric("PredictionConfidence", confidence)
        
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
# NEW AWS DATASET & RETRAINING ENDPOINTS
# ============================================

@app.post("/add-training-data")
async def add_training_data(data: TrainingDataInput):
    """
    Add verified custom training data directly to DynamoDB (is_training_sample = True)
    """
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
            detail="Failed to save training sample to DynamoDB."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/verify-prediction")
async def verify_prediction(data: VerifyLogInput):
    """
    Verify/correct a past prediction log and promote it to a training sample in DynamoDB
    """
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
    """
    Retrieve all prediction logs and verified training samples from DynamoDB
    """
    try:
        logs = aws_service.get_all_logs_from_dynamodb()
        # Sort logs by timestamp (descending)
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
    """
    Retrain the RandomForestClassifier on public data + custom DynamoDB verified training data,
    save the updated model and label encoder to S3, and hot-reload them in-memory.
    """
    global model, label_encoder
    try:
        print("🔄 Retraining pipeline initiated...")
        
        # 1. Download baseline dataset from public URL
        url = 'https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv'
        df = pd.read_csv(url)
        
        # 2. Preprocess baseline dataset
        # Clean tinggi_meter
        df['tinggi_meter'] = df['tinggi_meter'].astype(str).str.extract('(\d+)').astype(float)
        
        # Extract lat/lon from geolokasi
        import re
        def extract_coordinates(geolokasi_str):
            if pd.isna(geolokasi_str):
                return np.nan, np.nan
            match = re.search(r'([-+]?\d+\.?\d*)°([NS])\s+([-+]?\d+\.?\d*)°([EW])',
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
        df = df.drop(columns=['geolokasi'])
        df_cleaned = df.dropna().copy()
        
        # Keep features & shapes
        df_final = df_cleaned[['tinggi_meter', 'lat', 'lon', 'bentuk']].copy()
        
        # 3. Retrieve verified training data from DynamoDB
        db_items = aws_service.get_training_samples_from_dynamodb()
        print(f"📥 Found {len(db_items)} verified custom training samples in DynamoDB.")
        
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
            print("✅ Merged baseline data with custom DynamoDB samples.")
        
        # 4. Define features (X) and target (y)
        X = df_final[['tinggi_meter', 'lat', 'lon']]
        y = df_final['bentuk']
        
        # 5. Label Encode the shapes
        from sklearn.preprocessing import LabelEncoder
        new_label_encoder = LabelEncoder()
        y_encoded = new_label_encoder.fit_transform(y)
        
        # 6. Fit the RandomForestClassifier model
        from sklearn.ensemble import RandomForestClassifier
        new_model = RandomForestClassifier(random_state=42)
        new_model.fit(X, y_encoded)
        
        # Calculate training score
        score = float(new_model.score(X, y_encoded))
        
        # 7. Save model and label encoder files locally
        joblib.dump(new_model, 'volcano_classifier_model.joblib')
        joblib.dump(new_label_encoder, 'volcano_label_encoder.joblib')
        
        # 8. Upload files to S3
        aws_service.upload_model_to_s3('volcano_classifier_model.joblib', 'volcano_classifier_model.joblib')
        aws_service.upload_model_to_s3('volcano_label_encoder.joblib', 'volcano_label_encoder.joblib')
        
        # 9. Hot-reload model into FastAPI active state
        model = new_model
        label_encoder = new_label_encoder
        print("🚀 Model successfully retrained, saved to S3, and hot-reloaded into memory!")
        
        # 10. Generate classification report
        from sklearn.metrics import classification_report
        y_pred = model.predict(X)
        report = classification_report(y_encoded, y_pred, target_names=list(label_encoder.classes_), output_dict=True, zero_division=0)
        
        return {
            "success": True,
            "message": "Model retrained and hot-reloaded successfully.",
            "metrics": {
                "training_score": round(score, 4),
                "total_samples": len(df_final),
                "custom_samples_used": len(db_items),
                "classes": list(label_encoder.classes_)
            },
            "classification_report": report
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}"
        )