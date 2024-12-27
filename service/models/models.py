import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


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


    def fit(self, data: pd.DataFrame):
        """
        Обучает импутеры на тренировочных данных

        Parameters:
        data (pd.DataFrame): Тренировочные данные
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


    def select_heart_based_features(self, all_features) -> dict:
        return {
            'Age': all_features['Age'],
            'Sex': all_features['Sex'],
            'CheastPainType': all_features['CheastPainType'],
            'RestingBP': all_features['RestingBP'],
            'Cholesterol': all_features['Cholesterol'],
            'FastingBS': all_features['FastingBS'],
            'RestingECG': all_features['RestingECG'],
            'MaxHR': all_features['MaxHR'],
            'ExerciseAngina': all_features['ExerciseAngina'],
            'Oldpeak': all_features['Oldpeak'],
            'ST_Slope': all_features['ST_Slope'],
            'NumMajorVessels': all_features['NumMajorVessels'],
            'Thal': all_features['Thal']
        }


    def select_cardio_train_based_features(self, all_features) -> dict:
        return {
            'age': all_features['age'],
            'gender': all_features['gender'],
            'height': all_features['height'],
            'weight': all_features['weight'],
            'ap_hi': all_features['ap_hi'],
            'ap_lo': all_features['ap_lo'],
            'cholesterol': all_features['cholesterol'],
            'gluc': all_features['gluc'],
            'smoke': all_features['smoke'],
            'alco': all_features['alco'],
            'active': all_features['active']
        }


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