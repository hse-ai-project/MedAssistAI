from fastapi import APIRouter
from api.model import model_desc
from api.model_service.model_service import ModelService, setup_logger


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
