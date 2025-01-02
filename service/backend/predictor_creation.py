from api.model_classes.model_classes import HeartBasedPredictor, CardioTrainBasePredictor, PredictorComposer
import os
import pandas as pd
import joblib

# Создаем предикторы
heart_predictor = HeartBasedPredictor()
cardio_predictor = CardioTrainBasePredictor()

# Создаем композитную модель
composer = PredictorComposer(heart_predictor, cardio_predictor)

# Получаем пути к файлам        
heart_path = 'C:\\Users\\Daniil\\Desktop\\Python\\hse_project\\baseline\\ivan_chernykh_baseline_cml\\data\\heart.csv'
cardio_path = 'C:\\Users\\Daniil\\Desktop\\Python\\hse_project\\baseline\\daniil_vasilev_baseline_cml\\data\\cardio_train_correct.parquet'

# Загружаем данные
heart_data = pd.read_csv(heart_path)
cardio_data = pd.read_parquet(cardio_path)

# Подготавливаем данные для обучения
X1 = heart_data.drop('Target', axis=1).to_dict()
y1 = heart_data['Target']

X2 = cardio_data.drop(['cardio', 'id'], axis=1).to_dict()
y2 = cardio_data['cardio']

# Обучаем модель
composer.fit_splitted(X1, y1, X2, y2)

joblib.dump(composer, "model.pickle") 

print(type(joblib.load("model.pickle")))