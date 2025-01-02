import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

class HeartDataImputer:
    def __init__(self):
        self.features = {
            'numeric': ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'NumMajorVessels'],
            'categorical': ['Sex', 'CheastPainType', 'FastingBS', 'RestingECG',
                          'ExerciseAngina', 'ST_Slope', 'Thal']
        }

        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False
        self.scaler = StandardScaler()  # scaler

    def fit(self, data: pd.DataFrame):
        """
        Обучает импутеры на тренировочных данных. Также обучает StandardScaler и преобразовывает данные

        Parameters:
        data (pd.DataFrame): Тренировочные данные
        """
        data = pd.DataFrame(data)
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

        # Scaler
        self.scaler = self.scaler.fit(data[self.features['numeric']])
        
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
            if feature not in filled_data or filled_data[feature].isna().sum()!=0:
                filled_data[feature] = self.statistics['numeric'][feature]

        # Заполняем пропущенные категориальные значения
        for feature in self.features['categorical']:
            if feature not in filled_data or filled_data[feature].isna().sum()!=0:
                filled_data[feature] = self.statistics['categorical'][feature]
        # Scaler
        filled_data[self.features['numeric']] = self.scaler.transform(filled_data[self.features['numeric']])

        return filled_data
    

class HeartBasedPredictor:
    """Модель 1"""

    def __init__(self):
        self.imputer = HeartDataImputer()

        self.feature_order = [
            'Age', 'Sex', 'CheastPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope', 'NumMajorVessels', 'Thal'
        ]

        self.log_params =  {
                                'random_state': 42, 
                                'max_iter':     1000
                            }
        
        self.numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    def preprocess(self, features: pd.DataFrame) -> np.ndarray:
        return self.imputer.transform(features[self.feature_order])

    def fit(self, X: Dict[str, List[Any]], y: np.ndarray) -> 'HeartBasedPredictor':
        df = pd.DataFrame(X)[self.feature_order]
        self.imputer = self.imputer.fit(X)  # Сначала обучаем новый импутер на новых данных
        X_processed = self.preprocess(df)   # Преобразовываем данные на новом препроцессинге
        self.model = LogisticRegression(**self.log_params).fit(X_processed, y)
        return self

    def predict(self, features: Dict[str, Any]) -> float:
        X = self.preprocess(pd.DataFrame(features, index=[0]))
        return float(self.model.predict_proba(X)[0][1])
    

class CardioTrainBasePredictor:
    """Модель 2"""

    def __init__(self):
        self.fixed_features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        self.xgb_params =   {
                                'n_estimators':   500,
                                'learning_rate':  0.1,
                                'max_depth':      3,
                                'objective':      'binary:logistic',
                                'eval_metric':    'logloss',
                                'random_state':   12,
                                'n_splits':       4
                                
                            }        

    def preprocessing_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Функция, устанавливающая средние значения обучающего датафрейма в качестве значений для заполнения пропусков на предикте (если таковые будут присутствовать)

            Returns:
                X (pd.Dataframe) - подготовленный датафрейм для получения предсказаний
        """
        self.X_NaN_dict = df.mean().to_dict()    # Заполняем средним
        return self

    def preprocessing_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Функция предобработки входных данных. 
            В виду того, что данная модель является базовой, в данном варианте присутствуют только заполнения пропусков на случай, если какие-то столбцы будут не заполнены от пользователя

            Returns:
                X (pd.Dataframe) - подготовленный датафрейм для получения предсказаний
        """
        df = df[self.fixed_features]
        # Модуль заполнения пропусков
        for col in self.fixed_features:
            if (df[col].isna().sum() > 0) and (self.X_NaN_dict.get(col) != None):
                df[col] = self.X_NaN_dict.get(col)            
        return df
    
    def fit(self, X: Dict[str, List[Any]], y: List[int]):
        """Обучение модели"""
        self.preprocessing_fit(pd.DataFrame(X)[self.fixed_features])    # Получение готовых пропусков для заполнения в дальнейшем
        features = pd.DataFrame(X)[self.fixed_features]
        y = pd.DataFrame(y)
        # Инициализация KFold
        kf = KFold(n_splits=self.xgb_params.pop('n_splits', 4))

        for train_index, val_index in kf.split(features):
            X_train, X_val = features.iloc[train_index], features.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            self.model = XGBClassifier(**self.xgb_params).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        """Функция получения предсказаний из обработанного массива данных.

            Returns:    1 0
                        0.8 0.35
                np.array - массив numpy вероятностями заболевания пользователя, округлёнными до сотых.
        """ 
        df = self.preprocessing_transform(pd.DataFrame(X, index=[0]))
        return 1 / (1 + np.exp(-self.model.predict(df, output_margin=True)))    # Преобразование логита в условную вероятность
    

class PredictorComposer:
    """Общая модель"""

    def __init__(self, heart_based_predictor: HeartBasedPredictor, 
                 cardio_train_based_predictor: CardioTrainBasePredictor):
        self.heart_based_predictor = heart_based_predictor
        self.cardio_train_based_predictor = cardio_train_based_predictor

    def prepare_dict(self,  X: Dict[str, List[Any]]):
        X['age'] = X['Age']
        X['cholesterol'] = X['Cholesterol']

    def fit_splitted(self, X1: Dict[str, List[Any]], y1: np.ndarray, X2: Dict[str, List[Any]], y2: np.ndarray) -> 'PredictorComposer':
        self.heart_based_predictor.fit(X1, y1)
        self.cardio_train_based_predictor.fit(X2, y2)
        return self
    
    def fit(self, X: Dict[str, List[Any]], y: np.ndarray) -> 'PredictorComposer':
        self.prepare_dict(X)
        self.heart_based_predictor.fit(X, y)
        self.cardio_train_based_predictor.fit(X, y)
        return self
    
    def set_parameters(self, params: Dict[str, Dict[str, Any]]):
        if params is not None:
            if 'heart_based' in params:
                self.heart_based_predictor.log_params.update(**params['heart_based'])
            if 'cardio_based' in params:
                self.cardio_train_based_predictor.xgb_params.update(**params['cardio_based'])

    def predict(self, features: Dict[str, Any]) -> float:
        self.prepare_dict(features)
        heart_based_predict = self.heart_based_predictor.predict(features)
        cardio_train_based_predict = self.cardio_train_based_predictor.predict(features)
        return float((heart_based_predict + cardio_train_based_predict) / 2)