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


    def fit(self, data: Dict[str, List[Any]]):
        """
        Обучает импутеры на тренировочных данных

        Parameters:
        data (Dict[str, List[Any]]): Тренировочные данные
        """

        required_columns = self.features['numeric'] + self.features['categorical']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.numeric_imputer.fit(data[self.features['numeric']])
        self.categorical_imputer.fit(data[self.features['categorical']])

        self.statistics = {
            'numeric': data[self.features['numeric']].mean().to_dict(),
            'categorical': data[self.features['categorical']].mode().iloc[0].to_dict()
        }

        self.is_fitted = True
        return self


    def transform(self, patient_data: dict) -> dict:
        """
        Заполняет пропущенные значения в данных пациента

        Parameters:
        patient_data (dict): Данные пациента с возможными пропущенными значениями

        Returns:
        dict: Данные пациента с заполненными пропущенными значениями
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


    def preprocess(self, features):
        """
        Preprocesses input features

        Parameters:
        features (dict): Dictionary with patient features

        Returns:
        np.array: Preprocessed features array
        """

        features = self.imputer.transform(features)

        feature_order = [
            'Age', 'Sex', 'CheastPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope', 'NumMajorVessels', 'Thal'
        ]

        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

        X = np.array([[features[feature] for feature in feature_order]])
        numerical_indices = [feature_order.index(feat) for feat in numerical_features]
        X[:, numerical_indices] = self.scaler.transform(X[:, numerical_indices])

        return X
    
    def fit(self, X: Dict[str, List[Any]], y: np.array):
        """Обучение модели на выбранных признаках"""
        features_dict = self.select_features(X)
        X_processed = self.preprocess(features_dict)
        self.model.fit(X_processed, y)
        return self


    def predict(self, features):
        """
        Makes prediction for single patient

        Parameters:
        features (dict): Dictionary with patient features

        Returns:
        int: 0 for healthy, 1 for heart disease
        """
        X = self.preprocess(features)
        return self.model.predict_proba(X)[0][1]
    

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

    def __init__(self, heart_based_predictor, cardio_train_based_predictor):
        self.heart_based_predictor = heart_based_predictor
        self.cardio_train_based_predictor = cardio_train_based_predictor

    def fit(self, X: Dict[str, List[Any]], y: List[int]):
        """Обучение обеих моделей"""
        self.heart_based_predictor.fit(self.select_heart_based_features(X), y)
        self.cardio_train_based_predictor.fit(self.select_cardio_train_based_features(X), y)
        return self
    
    def set_parameters(self, params: Dict[str, Any]):
        """Установка параметров для обеих моделей"""
        if 'heart_based' in params:
            self.heart_based_predictor.model.set_params(**params['heart_based'])
        if 'cardio_based' in params:
            self.cardio_train_based_predictor.model.set_params(**params['cardio_based'])

    def select_heart_based_features(self, X) -> dict:
        features = [
            'Age', 'Sex', 'CheastPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope', 'NumMajorVessels', 'Thal'
        ]

        return {feature: X[feature] for feature in features}

    def select_cardio_train_based_features(self, X) -> dict:
        features = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]

        return {feature: X[feature] for feature in features}


    def predict(self, all_features):
        """
        Makes prediction for single patient

        Parameters:
        features (dict): Dictionary with patient features

        Returns:
        int: 0 for healthy, 1 for heart disease
        """

        heart_based_features = self.select_heart_based_features(all_features)
        heart_based_predict = self.heart_based_predictor.predict(heart_based_features)

        cardio_train_based_features = self.select_cardio_train_based_features(all_features)
        cardio_train_based_predict = self.cardio_train_based_predictor.predict(cardio_train_based_features)

        return (heart_based_predict + cardio_train_based_predict) / 2