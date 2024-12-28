from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Enum
from datetime import datetime

class Sex(int, Enum):
    FEMALE = 0
    MALE = 1

class ChestPainType(int, Enum):
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

class ExerciseAngina(int, Enum):
    YES = 1
    NO = 0

class ST_Slope(int, Enum):
    UPSLOPING = 0
    FLAT = 1 
    DOWNSLOPING = 2

class Thal(int, Enum):
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
    age: Optional[int] = Field(gt=0, default=None)
    sex: Optional[Sex] = None
    chest_pain_type: Optional[ChestPainType] = None
    resting_bp: Optional[int] = None
    cholesterol: Optional[int] = None
    fasting_bs: Optional[FastingBloodSugar] = None
    resting_ecg: Optional[RestingElectrocardiographic] = None
    max_hr: Optional[int] = None
    exercise_angina: Optional[ExerciseAngina] = None
    oldpeak: Optional[float] = None
    st_slope: Optional[ST_Slope] = None
    num_major_vessels: Optional[int] = None
    thal: Optional[Thal] = None
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
            if field_value is not None:
                if isinstance(field_value, Enum):
                    data[field_name] = field_value.value
                else:
                    data[field_name] = field_value
                    
        return data

class ModelInfo(BaseModel):
    id: str
    created_at: datetime
    is_active: bool
    parameters: Optional[Dict[str, any]] = None

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    active_model_id: Optional[str]

class TrainData(BaseModel):
    features: Dict[str, List[any]]
    target: List[int]

class FitRequest(BaseModel):
    id: str
    hyperparameters: Dict[str, any]
    train_data: TrainData
    timeout: Optional[int] = 10

class FitResponse(BaseModel):
    status: str
    message: str
    model_id: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_id: str

class SetModelRequest(BaseModel):
    model_id: str

class SetModelResponse(BaseModel):
    status: str
    previous_model_id: Optional[str]
    new_model_id: str
