from typing import List, Dict, Union

from fastapi import APIRouter

from pydantic import BaseModel, Field
from enum import Enum

router = APIRouter()


# Имплементация полей объектов
class Sex(Enum):
    "Пол"
    MALE = "Мужской"
    FEMALE = "Женский"


class ChestPainType(Enum):
    "Тип боли в груди"
    SELECT1 = "Бессимптомный"
    SELECT2 = "Типичная стенокардия"
    SELECT3 = "Атипичная стенокардия"
    SELECT4 = "Неангинозная боль"


class FastingBloodSugar(Enum):
    "Уровень сахара в крови натощак меньше 120 mg/d"
    NO = "Нет"
    YES = "Да"


class RestingElectrocardiographic(Enum):
    "Результаты электрокардиографии в состоянии покоя"
    SELECT1 = "Норма"
    SELECT2 = "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)"
    SELECT3 = "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса"


class ExerciseAngina:
    "Имеется стенокардия, вызванная физической нагрузкой"
    NO = "Нет"
    YES = "Да"


class ST_Slope(Enum):
    "Наклон сегмента ST при пиковой нагрузке"
    SELECT1 = "Плоский"
    SELECT2 = "Восходящий"
    SELECT3 = "Нисходящий"


class Thal(Enum):
    "Таллиевый стресс-тест"
    SELECT1 = "Нормальное значение"
    SELECT2 = "Ошибка"
    SELECT3 = "Фиксированный дефект"
    SELECT4 = "Обратимый дефект"


class Smoke(Enum):
    "Курите"
    NO = "Нет"
    YES = "Да"


class Alco(Enum):
    "Употребляете алкоголь"
    NO = "Нет"
    YES = "Да"


class Active(Enum):
    "Занимаетесь физической активностью"
    NO = "Нет"
    YES = "Да"


class PatientData(BaseModel):
    # фичи Вани
    Age: int = Field(gt=0)
    Sex: Sex
    CheastPainType: ChestPainType
    RestingBP: int
    Cholesterol: int
    FastingBS: FastingBloodSugar
    RestingECG: RestingElectrocardiographic
    MaxHR: int
    ExerciseAngina: ExerciseAngina
    Oldpeak: int
    ST_Slope: ST_Slope
    NumMajorVessels: int
    Thal: Thal

    # фичи Дани
    height: int = Field(gt=0)
    weight: int = Field(gt=0)
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: Smoke
    alco: Alco
    active: Active


class Verdict(Enum):
    SICK = "Болен"
    HEALTH = "Здоров"


class PatientPredict(BaseModel):
    verdict: Verdict


# API endpoints
@router.post("/predict", response_model=List[PatientPredict])
async def predict(patients: List[PatientData]) -> List[PatientPredict]:
    "Предсказание болен ли пациент или нет"
    for patient in patients:
        # some func here
        pass
    return [PatientPredict(verdict="Болен"), PatientPredict(verdict="Здоров")]


@router.post("/predict_proba", response_model=List[float])
async def predict(patients: List[PatientData]) -> List[float]:
    "Предсказание вероятности заболевания"
    for patient in patients:
        # some func here
        pass
    return [0.2, 0.1]


@router.get("/metrics", response_model=Dict[str, Union[int, float]])
async def predict() -> Dict[str, Union[int, float]]:
    "Получение метрик, получившихся во время обучения моделей"
    # some func here
    return {"roc-auc": 0.5, "accuracy": 0.7}
