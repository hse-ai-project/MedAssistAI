from fastapi import HTTPException
from typing import List, Dict, Any
import joblib
import multiprocessing
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import asyncio
from api.model.model import *
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger():
    '''Конфигурация логгера для последующей его использования'''
    logger_ = logging.getLogger("backend")
    if not logger_.hasHandlers():
        logger_.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler("logs/logs.log", when="D", interval=1)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger_.addHandler(handler)
    return logger_

def train_model(hyperparameters: Dict[str, Any], train_data: TrainData):
    """Training function to run in separate process"""
    try:
        with open('api/model_service/model.pickle', 'rb') as f:
            model = joblib.load(f)
            
        X_train = train_data.features
        y_train = train_data.target

        if hyperparameters:
            model.set_parameters(hyperparameters)
        
        model.fit(X_train, y_train)
        
        return model

    except Exception as e:
        raise Exception(f"Error in model training: {e}")

class ModelService:
    def __init__(self, max_workers: int = 3):
        self.models = {}
        self.active_model_id = None
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.load_initial_model()

    def load_initial_model(self):
        """Load pre-trained model on startup"""
        try:
            with open('api/model_service/model.pickle', 'rb') as f:
                model = joblib.load(f)
            model_info = ModelInfo(
                id="default_model",
                created_at=datetime.now(),
                is_active=True,
                parameters={}
            )
            
            self.models["default_model"] = {
                "model": model,
                "info": model_info
            }

            self.active_model_id = "default_model"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load Model error: {str(e)}") 

    async def fit_model(self, model_id: str, hyperparameters: Dict[str, Any], 
                 train_data: TrainData, timeout: int = 10) -> FitResponse:
        """Fit model with timeout"""
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor,
                train_model,
                hyperparameters,
                train_data
            )

            model = await asyncio.wait_for(future, timeout=timeout)

            model_info = ModelInfo(
                id=model_id,
                created_at=datetime.now(),
                is_active=False,
                parameters=hyperparameters
            )
                
            self.models[model_id] = {
                "model": model,
                "info": model_info
            }

            return FitResponse(
                status="success",
                message="Model trained successfully",
                model_id=model_id
            )

        except asyncio.TimeoutError:
            return FitResponse(
                status="timeout",
                message=f"Training exceeded timeout of {timeout} seconds"
            )
            

        except Exception as e:
            return FitResponse(
                status="error",
                message=f"Training failed: {str(e)}"
            )
        
    def predict_batch(self, patients: List[PatientData]) -> BatchPredictResponse:
        """
        Получение предсказаний для группы пациентов от активной модели
        """
        if not self.active_model_id or self.active_model_id not in self.models:
            raise HTTPException(status_code=404, detail="No active model found")
            
        model = self.models[self.active_model_id]["model"]
        try:
            predictions = []
            for patient in patients:
                probability = model.predict(patient.to_dict())
                predictions.append(
                    PatientPrediction(
                        prediction=round(probability),
                        probability=float(probability)
                    )
                )
            
            return BatchPredictResponse(
                model_id=self.active_model_id,
                predictions=predictions
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    def get_models(self) -> ModelsResponse:
        """Get list of available models"""
        return ModelsResponse(
            models=[model["info"] for model in self.models.values()],
            active_model_id=self.active_model_id
        )

    def set_active_model(self, model_id: str) -> SetModelResponse:
        """Set active model"""
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        previous_model_id = self.active_model_id
        self.active_model_id = model_id
        
        for mid, model_data in self.models.items():
            model_data["info"].is_active = (mid == model_id)
            
        return SetModelResponse(
            status="success",
            previous_model_id=previous_model_id,
            new_model_id=model_id
        )

    def update_model(self, model_id: str, train_data: TrainData) -> FitResponse:
        """Update existing model with new data"""
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
        try:
            X_new = pd.DataFrame(train_data.features)
            y_new = train_data.target
            
            model = self.models[model_id]["model"]
            model.fit(X_new, y_new)
            
            self.models[model_id]["model"] = model
            self.models[model_id]["info"].created_at = datetime.now()
            
            return FitResponse(
                status="success",
                message="Model updated successfully",
                model_id=model_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")
        