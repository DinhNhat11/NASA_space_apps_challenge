from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from io import BytesIO
import json
import uuid
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Exoplanet Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for datasets and training jobs
datasets = {}
training_jobs = {}
trained_models = {}

# Pretrained model info (simulated)
PRETRAINED_MODELS = {
    "kepler": {
        "name": "Kepler Model",
        "accuracy": 95.3,
        "mission": "Kepler",
        "trained_on": "Kepler mission dataset",
        "features": ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq"]
    },
    "k2": {
        "name": "K2 Model",
        "accuracy": 93.7,
        "mission": "K2",
        "trained_on": "K2 mission dataset",
        "features": ["k2_period", "k2_duration", "k2_depth", "k2_prad", "k2_teq"]
    },
    "tess": {
        "name": "TESS Model",
        "accuracy": 93.5,
        "mission": "TESS",
        "trained_on": "TESS mission dataset",
        "features": ["tic_period", "tic_duration", "tic_depth", "tic_prad", "tic_teq"]
    }
}

# Pydantic models
class TrainConfig(BaseModel):
    dataset_id: str
    algorithm: str = "random_forest"
    test_size: float = 0.2
    max_iterations: int = 100
    learning_rate: float = 0.01
    target_column: str = "disposition"
    feature_columns: Optional[List[str]] = None

class PredictRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    upload_time: str
    rows: int
    columns: int
    mission: Optional[str] = None

# Helper functions
def validate_file(file: UploadFile) -> str:
    """Validate file format"""
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif filename.endswith('.json'):
        return 'json'
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, XLSX, or JSON")

def load_dataframe(content: bytes, file_type: str) -> pd.DataFrame:
    """Load data into pandas DataFrame"""
    try:
        if file_type == 'csv':
            return pd.read_csv(BytesIO(content))
        elif file_type == 'excel':
            return pd.read_excel(BytesIO(content))
        elif file_type == 'json':
            return pd.read_json(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

def get_model(algorithm: str, config: TrainConfig):
    """Get ML model based on algorithm choice"""
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.max_iterations,
            random_state=42,
            n_jobs=-1
        )
    elif algorithm == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=config.max_iterations,
            learning_rate=config.learning_rate,
            random_state=42
        )
    elif algorithm == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=config.max_iterations,
            learning_rate_init=config.learning_rate,
            random_state=42
        )
    elif algorithm == "svm":
        return SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported algorithm")

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dataset statistics"""
    
    # Helper function to clean values
    def clean_value(val):
        if pd.isna(val) or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        return val
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Clean missing values count
    missing_by_column = {}
    for col in df.columns:
        missing_by_column[col] = int(df[col].isnull().sum())
    
    stats = {
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.columns) - len(numeric_cols),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        },
        "numeric_stats": {},
        "categorical_stats": {},
        "missing_by_column": missing_by_column,
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    
    # Numeric statistics with cleaning
    for col in numeric_cols[:10]:
        stats["numeric_stats"][col] = {
            "mean": clean_value(df[col].mean()),
            "median": clean_value(df[col].median()),
            "std": clean_value(df[col].std()),
            "min": clean_value(df[col].min()),
            "max": clean_value(df[col].max()),
            "quartiles": {
                "25%": clean_value(df[col].quantile(0.25)),
                "50%": clean_value(df[col].quantile(0.50)),
                "75%": clean_value(df[col].quantile(0.75))
            }
        }
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols[:5]:
        value_counts = df[col].value_counts().head(10).to_dict()
        stats["categorical_stats"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()}
        }
    
    return stats

def prepare_training_data(df: pd.DataFrame, config: TrainConfig):
    """Prepare data for training"""
    # Check if target column exists
    if config.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{config.target_column}' not found")
    
    # Get feature columns
    if config.feature_columns:
        feature_cols = [col for col in config.feature_columns if col in df.columns]
    else:
        # Auto-select numeric columns excluding target
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if config.target_column in feature_cols:
            feature_cols.remove(config.target_column)
    
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No valid feature columns found")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[config.target_column].copy()

    # Handle missing values - fill with column mean for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            col_mean = X[col].mean()
            # If mean is NaN (all values are NaN), fill with 0
            if pd.isna(col_mean):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(col_mean)
        else:
            X[col] = X[col].fillna(0)

    # Final safety check - fill any remaining NaNs with 0
    X = X.fillna(0)
    
    # Handle categorical target (encode if needed)
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        label_mapping = None
    
    return X, y, feature_cols, label_mapping

# API Endpoints

app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve index.html from root
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api")
async def root():
    return {"message": "Exoplanet Detection API", "version": "1.0.0"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), mission: Optional[str] = None):
    """Upload and validate dataset"""
    try:
        # Validate file type
        file_type = validate_file(file)
        
        # Read file content
        content = await file.read()
        
        # Load into DataFrame
        df = load_dataframe(content, file_type)
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Store dataset
        datasets[dataset_id] = {
            "dataframe": df,
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "mission": mission,
            "rows": len(df),
            "columns": len(df.columns)
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "mission": mission
            }
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    dataset_list = []
    for dataset_id, data in datasets.items():
        dataset_list.append({
            "dataset_id": dataset_id,
            "filename": data["filename"],
            "upload_time": data["upload_time"],
            "rows": data["rows"],
            "columns": data["columns"],
            "mission": data.get("mission")
        })
    return {"datasets": dataset_list}

@app.get("/explore/{dataset_id}")
async def explore_dataset(dataset_id: str):
    """Get detailed statistics and visualizations for a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["dataframe"]
    
    # Helper function to clean NaN values
    def clean_value(val):
        if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        return val
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Get sample data and clean it
    sample_data_raw = df.head(10).to_dict(orient='records')
    sample_data = []
    for row in sample_data_raw:
        cleaned_row = {k: clean_value(v) for k, v in row.items()}
        sample_data.append(cleaned_row)
    
    # Distribution data for visualizations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    distributions = {}
    
    for col in numeric_cols[:5]:  # First 5 numeric columns for histograms
        if not df[col].isnull().all():
            try:
                hist, bin_edges = np.histogram(df[col].dropna(), bins=20)
                distributions[col] = {
                    "counts": [int(x) for x in hist],
                    "bins": [float(x) for x in bin_edges]
                }
            except:
                continue
    
    # Correlation matrix for numeric columns
    correlation_matrix = None
    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr()
            # Clean correlation matrix
            corr_clean = corr.fillna(0).replace([np.inf, -np.inf], 0)
            correlation_matrix = {
                "columns": corr_clean.columns.tolist(),
                "data": [[clean_value(val) for val in row] for row in corr_clean.values.tolist()]
            }
        except:
            correlation_matrix = None
    
    return {
        "dataset_id": dataset_id,
        "statistics": stats,
        "sample_data": sample_data,
        "distributions": distributions,
        "correlation_matrix": correlation_matrix,
        "column_info": {
            "all_columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
    }

@app.post("/train")
async def train_model(config: TrainConfig, background_tasks: BackgroundTasks):
    """Train a new model"""
    if config.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    training_jobs[job_id] = {
        "status": "started",
        "progress": 0,
        "message": "Initializing training...",
        "config": config.dict(),
        "start_time": datetime.now().isoformat()
    }
    
    # Start training in background
    background_tasks.add_task(train_model_background, job_id, config)
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Training started"
    }

async def train_model_background(job_id: str, config: TrainConfig):
    """Background task for model training"""
    try:
        # Update status
        training_jobs[job_id]["status"] = "loading"
        training_jobs[job_id]["progress"] = 10
        training_jobs[job_id]["message"] = "Loading dataset..."
        
        df = datasets[config.dataset_id]["dataframe"]
        
        # Prepare data
        training_jobs[job_id]["progress"] = 20
        training_jobs[job_id]["message"] = "Preprocessing data..."
        
        X, y, feature_cols, label_mapping = prepare_training_data(df, config)
        
        # Split data
        training_jobs[job_id]["progress"] = 30
        training_jobs[job_id]["message"] = "Splitting train/test sets..."
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Replace any NaN or Inf values that may have been created by scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train model
        training_jobs[job_id]["progress"] = 50
        training_jobs[job_id]["message"] = "Training model..."
        
        model = get_model(config.algorithm, config)
        model.fit(X_train_scaled, y_train)
        
        # Validate
        training_jobs[job_id]["progress"] = 80
        training_jobs[job_id]["message"] = "Validating results..."
        
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store model results
        model_id = str(uuid.uuid4())
        trained_models[model_id] = {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "label_mapping": label_mapping,
            "algorithm": config.algorithm,
            "trained_on": config.dataset_id,
            "train_time": datetime.now().isoformat()
        }
        
        # Complete training
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["message"] = "Training complete!"
        training_jobs[job_id]["model_id"] = model_id
        training_jobs[job_id]["results"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "test_size": len(X_test),
            "train_size": len(X_train),
            "features_used": feature_cols
        }
        training_jobs[job_id]["end_time"] = datetime.now().isoformat()
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        training_jobs[job_id]["error"] = str(e)
        


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@app.get("/models")
async def list_models():
    """List all available models (pretrained + trained)"""
    models_list = []
    
    # Add pretrained models
    for model_id, model_info in PRETRAINED_MODELS.items():
        models_list.append({
            "model_id": model_id,
            "type": "pretrained",
            **model_info
        })
    
    # Add trained models
    for model_id, model_data in trained_models.items():
        models_list.append({
            "model_id": model_id,
            "type": "custom",
            "name": f"Custom {model_data['algorithm'].replace('_', ' ').title()} Model",
            "algorithm": model_data['algorithm'],
            "trained_on": model_data['trained_on'],
            "train_time": model_data['train_time']
        })
    
    return {"models": models_list}

@app.post("/predict")
async def predict(request: PredictRequest):
    """Make predictions using a model"""
    model_id = request.model_id
    
    # Check if it's a pretrained model (simulated prediction)
    if model_id in PRETRAINED_MODELS:
        # Simulate prediction for pretrained models
        prediction = np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'])
        confidence = np.random.uniform(0.7, 0.99)
        
        return {
            "success": True,
            "prediction": prediction,
            "confidence": float(confidence),
            "model": PRETRAINED_MODELS[model_id]["name"],
            "note": "This is a simulated prediction from pretrained model"
        }
    
    # Check if it's a trained model
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model_data = trained_models[model_id]
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_cols = model_data["feature_columns"]
        
        # Prepare input data
        input_df = pd.DataFrame([request.data])
        
        # Check if all required features are present
        missing_features = set(feature_cols) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)}"
            )
        
        # Select and order features
        X = input_df[feature_cols]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Convert prediction back to label if needed
        if model_data["label_mapping"]:
            inv_mapping = {v: k for k, v in model_data["label_mapping"].items()}
            prediction_label = inv_mapping.get(prediction, str(prediction))
        else:
            prediction_label = str(prediction)
        
        result = {
            "success": True,
            "prediction": prediction_label,
            "model_id": model_id,
            "algorithm": model_data["algorithm"]
        }
        
        if probabilities is not None:
            result["confidence"] = float(max(probabilities))
            result["probabilities"] = probabilities.tolist()
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(model_id: str, dataset_id: str):
    """Make batch predictions on an entire dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if model_id not in trained_models and model_id not in PRETRAINED_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    df = datasets[dataset_id]["dataframe"]
    
    # For pretrained models, simulate predictions
    if model_id in PRETRAINED_MODELS:
        predictions = np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'], size=len(df))
        confidences = np.random.uniform(0.7, 0.99, size=len(df))
        
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                "row_index": i,
                "prediction": pred,
                "confidence": float(conf)
            })
        
        return {
            "success": True,
            "total_predictions": len(results),
            "predictions": results[:100],  # Return first 100
            "note": "Simulated predictions from pretrained model"
        }
    
    # For trained models, make actual predictions
    try:
        model_data = trained_models[model_id]
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_cols = model_data["feature_columns"]
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            if model_data["label_mapping"]:
                inv_mapping = {v: k for k, v in model_data["label_mapping"].items()}
                pred_label = inv_mapping.get(pred, str(pred))
            else:
                pred_label = str(pred)
            
            result = {
                "row_index": i,
                "prediction": pred_label
            }
            
            if probabilities is not None:
                result["confidence"] = float(max(probabilities[i]))
            
            results.append(result)
        
        # Calculate distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            "success": True,
            "total_predictions": len(results),
            "predictions": results[:100],  # Return first 100
            "distribution": distribution
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    del datasets[dataset_id]
    return {"success": True, "message": "Dataset deleted"}

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del trained_models[model_id]
    return {"success": True, "message": "Model deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
