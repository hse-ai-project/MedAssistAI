import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.impute import SimpleImputer

class HeartDataImputer:
    """Класс для заполнения пропусков для модели Heart"""

    def __init__(self):
        self.features = {
            'numeric': ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'NumMajorVessels'],
            'categorical': ['Sex', 'CheastPainType', 'FastingBS', 'RestingECG',
                          'ExerciseAngina', 'ST_Slope', 'Thal']
        }

        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False


    def fit(self, data: Dict[str, List[Any]]) -> 'HeartDataImputer':
        required_columns = self.features['numeric'] + self.features['categorical']
        missing_columns = set(required_columns) - set(data.keys())
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = pd.DataFrame(data)
        self.numeric_imputer.fit(df[self.features['numeric']])
        self.categorical_imputer.fit(df[self.features['categorical']])

        self.statistics = {
            'numeric': df[self.features['numeric']].mean().to_dict(),
            'categorical': df[self.features['categorical']].mode().iloc[0].to_dict()
        }
        self.is_fitted = True
        return self


    def transform(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Заполняет пропущенные значения в данных пациента

        Parameters:
        patient_data (Dict[str, Any]): Данные пациента с возможными пропущенными значениями

        Returns:
        Dict[str, Any]: Данные пациента с заполненными пропущенными значениями
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")

        filled_data = patient_data.copy()

        # Заполняем пропущенные числовые значения
        for feature in self.features['numeric']:
            if feature not in filled_data or filled_data[feature] is None:
                filled_data[feature] = self.statistics['numeric'][feature]

        # Заполняем пропущенные категориальные значения
        for feature in self.features['categorical']:
            if feature not in filled_data or filled_data[feature] is None:
                filled_data[feature] = self.statistics['categorical'][feature]

        return filled_data
    

class HeartBasedPredictor:
    """Модель 1"""

    def __init__(self, model, scaler, imputer):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer

        self.feature_order = [
            'Age', 'Sex', 'CheastPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope', 'NumMajorVessels', 'Thal'
        ]
        
        self.numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


    def preprocess(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_order]
        X = self.imputer.transform(X)
        X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X.values

    def fit(self, X: Dict[str, List[Any]], y: np.ndarray) -> 'HeartBasedPredictor':
        df = pd.DataFrame(X)
        X_processed = self.preprocess(df)
        self.model.fit(X_processed, y)
        return self

    def predict(self, features: Dict[str, Any]) -> float:
        df = pd.DataFrame([features])
        X = self.preprocess(df)
        return float(self.model.predict_proba(X)[0][1])
    

class CardioTrainBasePredictor:
    """Модель 2"""

    def __init__(self, model_path, X_NaN_dict):
        """Инициализация класса

            Parameters:
                X - входной массив данных, на котором получаются предсказания
                model_path - модель, на основе которой будут получатся предсказания
                X_NaN_dict - словарь для заполнения пропусков во входном массиве данных
        """
        self.X_NaN_dict = X_NaN_dict
        self.model = model_path


    def preprocessing(self):
        """
            Функция предобработки входных данных. 
            В виду того, что данная модель является базовой, в данном варианте присутствуют только заполнения пропусков на случай, если какие-то столбцы будут не заполнены от пользователя

            Returns:
                X (pd.Dataframe) - подготовленный датафрейм для получения предсказаний
        """
        # Модуль заполнения пропусков
        for col in self.model.feature_names_in_:
            if (self.X[col].isna().sum() > 0) and (self.X_NaN_dict.get(col) != None):
                self.X[col] = self.X_NaN_dict.get(col)
        return self.X
    
    def fit(self, X: Dict[str, List[Any]], y: List[int]):
        """Обучение модели"""
        df = pd.DataFrame(X)
        self.model.fit(df, y)
        return self

    def predict(self, X):
        """Функция получения предсказаний из обработанного массива данных.

            Returns:    1 0
                        0.8 0.35
                np.array - массив numpy вероятностями заболевания пользователя, округлёнными до сотых.
        """
        self.X = pd.DataFrame(X, index=[0])
        self.X_prepared = self.preprocessing()
        return 1 / (1 + np.exp(-self.model.predict(self.X, output_margin=True))) # Преобразование логита в условную вероятность
    

class PredictorComposer:
    """Общая модель"""

    def __init__(self, heart_based_predictor: HeartBasedPredictor, 
                 cardio_train_based_predictor: CardioTrainBasePredictor):
        self.heart_based_predictor = heart_based_predictor
        self.cardio_train_based_predictor = cardio_train_based_predictor

    def fit(self, X: Dict[str, List[Any]], y: np.ndarray) -> 'PredictorComposer':
        self.heart_based_predictor.fit(X, y)
        self.cardio_train_based_predictor.fit(X, y)
        return self
    
    def set_parameters(self, params: Dict[str, Dict[str, Any]]) -> None:
        if 'heart_based' in params:
            self.heart_based_predictor.model.set_params(**params['heart_based'])
        if 'cardio_based' in params:
            self.cardio_train_based_predictor.model.set_params(**params['cardio_based'])

    def predict(self, features: Dict[str, Any]) -> float:
        heart_based_predict = self.heart_based_predictor.predict(features)
        cardio_train_based_predict = self.cardio_train_based_predictor.predict(features)
        return float((heart_based_predict + cardio_train_based_predict) / 2)