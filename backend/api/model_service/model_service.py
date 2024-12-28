from fastapi import HTTPException
from typing import List, Dict
import pickle
import multiprocessing
from datetime import datetime
import pandas as pd
from model.model import *

def train_model(hyperparameters: Dict[str, any], train_data: TrainData, queue: multiprocessing.Queue):
    """Training function to run in separate process"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        X_train = train_data.features
        y_train = train_data.target

        if hyperparameters:
            model.set_parameters(hyperparameters)
        
        model.fit(X_train, y_train)
        
        queue.put(model)
    except Exception as e:
        queue.put(e)

class ModelService:
    def __init__(self):
        self.models = {}
        self.active_model_id = None
        self.load_initial_model()

    def load_initial_model(self):
        """Load pre-trained model on startup"""
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            
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
            print(f"Error loading initial model: {e}")

    def fit_model(self, model_id: str, hyperparameters: Dict[str, any], 
                 train_data: TrainData, timeout: int = 10) -> FitResponse:
        """Fit model with timeout"""
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(
            target=train_model, 
            args=(hyperparameters, train_data, queue)
        )
        
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return FitResponse(
                status="timeout",
                message=f"Training exceeded timeout of {timeout} seconds"
            )
        
        try:
            result = queue.get_nowait()
            if isinstance(result, Exception):
                raise result
                
            model_info = ModelInfo(
                id=model_id,
                created_at=datetime.now(),
                is_active=False,
                parameters=hyperparameters
            )
            
            self.models[model_id] = {
                "model": result,
                "info": model_info
            }
            
            return FitResponse(
                status="success",
                message="Model trained successfully",
                model_id=model_id
            )
            
        except Exception as e:
            return FitResponse(
                status="error",
                message=f"Training failed: {str(e)}"
            )

    def predict(self, data: PatientData) -> PredictResponse:
        """Make prediction using active model"""
        if not self.active_model_id or self.active_model_id not in self.models:
            raise HTTPException(status_code=404, detail="No active model found")
            
        model = self.models[self.active_model_id]["model"]
        
        try:
            features = data.to_dict()
            probability = model.predict(features)
            prediction = round(probability)
            
            return PredictResponse(
                prediction=round(prediction),
                probability=float(probability),
                model_id=self.active_model_id
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
        