import os
import logging
from typing import Dict, Any
import torch
import numpy as np
from llm_models import tokenizer, tokenize_dp, MultiOutputModel, SymptomExtractionModel

logger = logging.getLogger("backend")

class TextProcessor:    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_ready = False
        
        required_files = [
            "symptom_extraction_model.pt",
            "model_deeppavlov_tokens.pt"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей"""
        try:
            logger.info("Loading text processing models...")
            
            torch.set_num_threads(2)
            
            self.output_config = [
                ("thal", "categorical", 4), ("gluc", "categorical", 2),
                ("ap_lo", "numerical", 1), ("ap_hi", "numerical", 1),
                ("active", "categorical", 2), ("st_slope", "categorical", 3),
                ("max_hr", "numerical", 1), ("smoke", "categorical", 2),
                ("resting_bp", "numerical", 1), ("oldpeak", "categorical", 2),
                ("cholesterol_int", "numerical", 1), ("gender", "categorical", 2),
                ("cholesterol", "categorical", 2), ("fastingbs", "categorical", 2),
                ("exerciseangina", "categorical", 2), ("restingecg", "categorical", 3),
                ("age", "numerical", 1), ("alco", "categorical", 2),
                ("cheastpaintype", "categorical", 4), ("nummajorvessels", "numerical", 1),
                ("weight", "numerical", 1), ("height", "numerical", 1),
            ]
            
            self.numerical_features = [
                "age", "restingbp", "ap_hi", "weight", "cholesterol_int",
                "height", "maxhr", "ap_lo"
            ]
            
            self.categorical_features = [
                "gluc", "restingecg", "fastingbs", "cholesterol", "smoke",
                "oldpeak", "gender", "thal", "exerciseangina", "nummajorvessels",
                "active", "cheastpaintype", "alco", "st_slope"
            ]
            
            self.numerical_ranges = {
                "ap_lo": (40, 150), "ap_hi": (60, 240), "maxhr": (70, 240),
                "restingbp": (70, 230), "cholesterol_int": (126, 565),
                "age": (18, 101), "nummajorvessels": (0, 5),
                "weight": (40, 201), "height": (150, 221),
            }
            
            self.numerical_scalers = {
                "age": {"mean": 58.82342342342342, "std": 24.118905727777065},
                "restingbp": {"mean": 148.4099017109574, "std": 46.36705005876055},
                "ap_hi": {"mean": 150.31573075492747, "std": 52.51481911869445},
                "weight": {"mean": 117.81586608442504, "std": 46.31761287129987},
                "cholesterol_int": {"mean": 343.5028797696184, "std": 127.67927228795011},
                "height": {"mean": 185.68543407192155, "std": 20.131510829104766},
                "maxhr": {"mean": 154.69140337986775, "std": 49.199315639371164},
                "ap_lo": {"mean": 94.87477702461648, "std": 31.179277714643494},
            }
            
            self.categorical_mapping = {
                "gluc": [0.0, 1.0, "NaN"], "restingecg": [0.0, 1.0, 2.0, "NaN"],
                "fastingbs": [0.0, 1.0, "NaN"], "cholesterol": [0.0, 1.0, "NaN"],
                "smoke": [0.0, 1.0, "NaN"], "oldpeak": [0.0, 1.0, "NaN"],
                "gender": [0.0, 1.0, "NaN"], "thal": [0.0, 1.0, 2.0, 3.0, "NaN"],
                "exerciseangina": [0.0, 1.0, "NaN"], "nummajorvessels": [0.0, 1.0, 2.0, 3.0, 4.0, "NaN"],
                "active": [0.0, 1.0, "NaN"], "cheastpaintype": [0.0, 1.0, 2.0, 3.0, "NaN"],
                "alco": [0.0, 1.0, "NaN"], "st_slope": [0.0, 1.0, 2.0, "NaN"],
            }
            
            self.categorical_num_classes = {
                "gluc": 3, "restingecg": 4, "fastingbs": 3, "cholesterol": 3,
                "smoke": 3, "oldpeak": 3, "gender": 3, "thal": 5,
                "exerciseangina": 3, "nummajorvessels": 6, "active": 3,
                "cheastpaintype": 5, "alco": 3, "st_slope": 4,
            }
            
            model_state_dict = torch.load(
                "model_deeppavlov_tokens.pt",
                map_location=self.device,
                weights_only=False
            )
            self.model_one = MultiOutputModel(
                embedding_dim=375,
                hidden_dim=512,
                output_config=self.output_config
            )
            self.model_one.load_state_dict(model_state_dict)
            self.model_one.to(self.device)
            self.model_one.eval()
            
            for param in self.model_one.parameters():
                param.requires_grad = False
            
            model_state_dict = torch.load(
                "symptom_extraction_model.pt",
                map_location=self.device,
                weights_only=False
            )
            self.predict_model = SymptomExtractionModel(
                numerical_features=self.numerical_features,
                categorical_features=self.categorical_features,
                categorical_num_classes=self.categorical_num_classes,
            )
            self.predict_model.load_state_dict(model_state_dict)
            self.predict_model.to(self.device)
            self.predict_model.eval()
            
            for param in self.predict_model.parameters():
                param.requires_grad = False
            
            self.tokenizer = tokenizer
            self.tokenize_dp = tokenize_dp
            
            self.models_ready = True
            logger.info("Text processing models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Cannot initialize text processor: {e}")
    
    def extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """
        Извлекает признаки из текста и возвращает в формате для ML модели
        
        Args:
            text: Описание симптомов
            
        Returns:
            Словарь с признаками в формате PatientData
        """
        if not self.models_ready:
            raise RuntimeError("Text processor not ready")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            with torch.no_grad():
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                numerical_outputs, numerical_missing_outputs, categorical_outputs = \
                    self.predict_model(**encoding)
                
                inputs = self.tokenize_dp(text).unsqueeze(0).to(self.device)
                outputs = self.model_one(inputs).squeeze(0).cpu()
                
                pred_sanya = {}
                start_idx = 0
                for name, _, num_classes in self.output_config:
                    pred_sanya[name] = outputs[start_idx:start_idx + num_classes]
                    start_idx += num_classes
                
                pred_sanya["maxhr"] = pred_sanya["max_hr"]
                pred_sanya["restingbp"] = pred_sanya["resting_bp"]
                
                predictions = {}
                for feature in self.numerical_features:
                    missing_probs = torch.softmax(numerical_missing_outputs[feature], dim=1)
                    is_missing = missing_probs[0, 1] > 0.5
                    
                    if is_missing:
                        predictions[feature] = None
                    else:
                        value_danya = numerical_outputs[feature].cpu().numpy()[0][0]
                        value_sanya = pred_sanya[feature].cpu().numpy()[0]
                        mean_val = (value_danya + value_sanya) / 2

                        if feature in self.numerical_scalers:
                            scaler = self.numerical_scalers[feature]
                            mean_val = mean_val * scaler["std"] + scaler["mean"]
                        
                        if feature in self.numerical_ranges:
                            min_val, max_val = self.numerical_ranges[feature]
                            mean_val = max(min_val, min(max_val, mean_val))
                        
                        predictions[feature] = round(float(mean_val), 1)
                
                for feature in self.categorical_features:
                    probabilities = torch.softmax(categorical_outputs[feature], dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
                    class_values = self.categorical_mapping[feature]
                    
                    if predicted_class_idx == len(class_values) - 1:
                        predictions[feature] = None
                    else:
                        value_sanya = pred_sanya[feature].cpu().numpy()
                        value_danya = categorical_outputs[feature].cpu().numpy()[0][:-1]
                        mean_val = np.mean((value_sanya, value_danya), axis=0)
                        class_idx = np.argmax(mean_val)
                        
                        predictions[feature] = float(class_values[class_idx])
                
                return self._convert_to_patient_format(predictions)
                
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise
    
    def _convert_to_patient_format(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразует предсказания в формат PatientData"""
        
        field_mapping = {
            'age': 'Age',
            'gender': 'Sex',
            'cheastpaintype': 'CheastPainType',
            'restingbp': 'RestingBP',
            'cholesterol': 'Cholesterol',
            'fastingbs': 'FastingBS',
            'restingecg': 'RestingECG',
            'maxhr': 'MaxHR',
            'exerciseangina': 'ExerciseAngina',
            'oldpeak': 'Oldpeak',
            'st_slope': 'ST_Slope',
            'nummajorvessels': 'NumMajorVessels',
            'thal': 'Thal',
            'height': 'height',
            'weight': 'weight',
            'ap_hi': 'ap_hi',
            'ap_lo': 'ap_lo',
            'gluc': 'gluc',
            'smoke': 'smoke',
            'alco': 'alco',
            'active': 'active'
        }
        
        patient_data = {}
        for old_key, new_key in field_mapping.items():
            if old_key in predictions:
                value = predictions[old_key]
                
                if new_key in ['Age', 'RestingBP', 'MaxHR', 'NumMajorVessels', 
                              'height', 'weight', 'ap_hi', 'ap_lo'] and value is not None:
                    patient_data[new_key] = int(value)
                elif new_key == 'Oldpeak' and value is not None:
                    patient_data[new_key] = float(value)
                elif value is not None:
                    patient_data[new_key] = int(value)
                else:
                    patient_data[new_key] = None
        
        if 'cholesterol_int' in predictions and predictions['cholesterol_int'] is not None:
            patient_data['Cholesterol'] = int(predictions['cholesterol_int'])
        
        return patient_data


text_processor = TextProcessor()