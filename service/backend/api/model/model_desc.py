from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class Sex_(int, Enum):
    FEMALE = 0
    MALE = 1


class ChestPainType_(int, Enum):
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3


class FastingBloodSugar(int, Enum):
    NORMAL = 0
    HIGH = 1


class RestingElectrocardiographic(int, Enum):
    NORMAL = 0
    ABNORMAL = 1
    HYPERTROPHY = 2


class ExerciseAngina_(int, Enum):
    YES = 1
    NO = 0


class ST_Slope_(int, Enum):
    UPSLOPING = 0
    FLAT = 1
    DOWNSLOPING = 2


class Thal_(int, Enum):
    NORMAL = 0
    FIXED_DEFECT = 1
    REVERSIBLE_DEFECT = 2


class Smoke(int, Enum):
    NO = 0
    YES = 1


class Alco(int, Enum):
    NO = 0
    YES = 1


class Active(int, Enum):
    NO = 0
    YES = 1


class PatientData(BaseModel):
    Age: Optional[int] = Field(gt=0, default=None)
    Sex: Optional[Sex_] = None
    CheastPainType: Optional[ChestPainType_] = None
    RestingBP: Optional[int] = None
    Cholesterol: Optional[int] = None
    FastingBS: Optional[FastingBloodSugar] = None
    RestingECG: Optional[RestingElectrocardiographic] = None
    MaxHR: Optional[int] = None
    ExerciseAngina: Optional[ExerciseAngina_] = None
    Oldpeak: Optional[float] = None
    ST_Slope: Optional[ST_Slope_] = None
    NumMajorVessels: Optional[int] = None
    Thal: Optional[Thal_] = None
    age: Optional[int] = Field(gt=0, default=None)
    gender: Optional[Sex_] = None
    cholesterol: Optional[int] = None
    height: Optional[int] = Field(gt=0, default=None)
    weight: Optional[int] = Field(gt=0, default=None)
    ap_hi: Optional[int] = None
    ap_lo: Optional[int] = None
    gluc: Optional[int] = None
    smoke: Optional[Smoke] = None
    alco: Optional[Alco] = None
    active: Optional[Active] = None

    def to_dict(self) -> dict:
        """
        Converts all PatientData fields to dictionary format

        Returns:
            dict: All fields in dictionary format
        """
        data = {}

        for field_name, field_value in self:
            if isinstance(field_value, Enum):
                data[field_name] = field_value.value
            else:
                data[field_name] = field_value

        return data


class ModelInfo(BaseModel):
    id: str
    created_at: datetime
    is_active: bool
    parameters: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    active_model_id: Optional[str]


class TrainData(BaseModel):
    features: Dict[str, List[Any]]
    target: List[int]


class FitRequest(BaseModel):
    id: str
    hyperparameters: Dict[str, Any]
    train_data: TrainData
    timeout: Optional[int] = 10


class FitResponse(BaseModel):
    status: str
    message: str
    model_id: Optional[str] = None


class BatchPredictRequest(BaseModel):
    patients: List[PatientData]


class PatientPrediction(BaseModel):
    prediction: int
    probability: float


class BatchPredictResponse(BaseModel):
    model_id: str
    predictions: List[PatientPrediction]


class SetModelRequest(BaseModel):
    model_id: str


class SetModelResponse(BaseModel):
    status: str
    previous_model_id: Optional[str]
    new_model_id: str


class TextToPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class TextToPredictionResponse(BaseModel):
    status: str
    prediction: Optional[PatientPrediction] = None
    message: Optional[str] = None