from fastapi import APIRouter, HTTPException
from api.model import model_desc
from api.model_service.model_service import ModelService, setup_logger
import time
from api.text_processor import text_processor


router = APIRouter()
model_service = ModelService()
logger = setup_logger()


@router.post("/fit", response_model=model_desc.FitResponse)
async def fit(request: model_desc.FitRequest):
    """
    Обучение новой модели

    Args:
        request: FitRequest с id, гиперпараметрами и тренировочными данными
    """
    logger.debug("Got request from client to /fit")
    return await model_service.fit_model(
        request.id,
        request.hyperparameters,
        request.train_data,
        request.timeout
    )


@router.post("/predict", response_model=model_desc.BatchPredictResponse)
async def predict(
    data: model_desc.BatchPredictRequest,
) -> model_desc.BatchPredictResponse:
    logger.debug("Got request from client to /predict")
    return model_service.predict_batch(data.patients)


@router.get("/models", response_model=model_desc.ModelsResponse)
async def get_models():
    """
    Получение списка всех доступных моделей и информации о них
    """
    logger.debug("Got request from client to /models")
    return model_service.get_models()


@router.post("/set_model", response_model=model_desc.SetModelResponse)
async def set_model(request: model_desc.SetModelRequest):
    """
    Установка активной модели по id

    Args:
        request: SetModelRequest с id модели для активации
    """
    logger.debug("Got request from client to /set_model")
    return model_service.set_active_model(request.model_id)


@router.post("/update_model/{model_id}", response_model=model_desc.FitResponse)
async def update_model(model_id: str, train_data: model_desc.TrainData):
    """
    Обновление существующей модели новыми данными

    Args:
        model_id: ID модели для обновления
        train_data: Новые тренировочные данные
    """
    logger.debug("Got request from client to /update_model/%s", model_id)
    return model_service.update_model(model_id, train_data)


@router.post("/predict_from_text", response_model=model_desc.TextToPredictionResponse)
async def predict_from_text(request: model_desc.TextToPredictionRequest):
    """
    Полный пайплайн: текст → предсказание
    
    Извлекает медицинские признаки из текста и делает предсказание риска
    """
    logger.debug("Got request from client to /predict_from_text")
    
    try:
        extracted_features = text_processor.extract_features_from_text(request.text)
        patient_data = model_desc.PatientData(**extracted_features)
        
        if not model_service.active_model_id or model_service.active_model_id not in model_service.models:
            raise HTTPException(status_code=404, detail="No active model found")
        
        model = model_service.models[model_service.active_model_id]["model"]
        probability = model.predict(patient_data.to_dict())
        
        prediction = model_desc.PatientPrediction(
            prediction=round(probability),
            probability=float(probability)
        )
        
        return model_desc.TextToPredictionResponse(
            status="success",
            prediction=prediction
        )
        
    except ValueError as e:
        return model_desc.TextToPredictionResponse(
            status="validation_error",
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Error in text-to-prediction: {e}")
        return model_desc.TextToPredictionResponse(
            status="error",
            message=f"Processing failed: {str(e)}"
        )