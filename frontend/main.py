import logging
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from logging.handlers import TimedRotatingFileHandler # RotatingFileHandler

def setup_logger():
    logger = logging.getLogger('frontend')
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG) 
        handler = TimedRotatingFileHandler('logs/logs.log', when='D', interval=1)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

st.write(
        """
        # Предсказание наличия сердечных заболевания
        Определяем, вероятность сердечных заболеваний на основе информации о пациенте (пациентах)
        """
    )

def process_main_page():
    '''
    Рендеринг основных элементов страницы (заголовки, изображения, sidebar, кнопки)
    '''
    image = Image.open('data/ded.jpg')
    st.image(image)
    data = process_side_bar_inputs()
    if st.button("Предсказать", key='predict'):
        # ручка /predict
        logger.debug("Predict button clicked")
        try:
            logger.debug("Attempting API call to /predict")
            prediction = requests.post("http://127.0.0.1:8000/model/predict", json=data).json()
            logger.debug("Got response from API for /predict")
            results = ["У вас выявлено возможное сердечное заболевание! Срочно обратитесь к врачу!" if elem == 1 else "У вас не выявлено сердечных заболеваний!" for elem in prediction]
        except Exception as err:
            logger.error("Cant get response from API for /predict_proba: " + str(err))
            results = ['Не удалось выполнить запрос к API']
        write_prediction(results)

    if st.button("Предсказать вероятности", key='predict_proba'):
        # ручка /predict_proba
        logger.debug("Predict proba button clicked")
        try:
            logger.debug("Attempting API call to /predict_proba")
            prediction_probs = requests.post("http://127.0.0.1:8000/model/predict_proba", json=data).json()
            logger.debug("Got response from API for /predict_proba")
        except Exception as err:
            logger.error("Cant get response from API for /predict_proba: " + str(err))
            prediction_probs = ['Не удалось выполнить запрос к API']
        write_prediction_proba(prediction_probs)

    if st.button("Метрики качества модели", key='metrics'):
        # ручка /metrics
        logger.debug("Metrics button clicked")
        try:
            logger.debug("Attempting API call to /metrics")
            metrics = requests.get('http://127.0.0.1:8000/model/metrics').json()
            logger.debug("Got response from API for /metrics")
        except Exception as err:
            logger.error("Cant get response from API for /predict_proba: " + str(err))
            metrics = ['Не удалось выполнить запрос к API']
        write_metrics(metrics)

    if st.button("Участники проекта", key='participants'):
        # ручка /participants
        logger.debug("Participants button clicked")
        try:
            logger.debug("Attempting API call to /participants")
            participants = requests.get('http://127.0.0.1:8000/participants').json()['status']
            logger.debug("Got response from API for /participants")
        except Exception as err:
            logger.error("Cant get response from API for /predict_proba: " + str(err))
            participants = 'Не удалось выполнить запрос к API'
        
        write_participants(participants)


# Ниже представлены функции для отрисовки полученных данных от FastAPI
def write_prediction(prediction):
    st.write("## Поставленный диагноз")
    st.table(pd.DataFrame(prediction, columns=['Диагноз']))
    logger.debug('Write result for /predict')

def write_prediction_proba(prediction_probs):
    st.write("## Вероятность поставленного диагноза")
    st.table(pd.DataFrame(prediction_probs, columns=['Вероятность сердечного заболевания']))
    logger.debug('Write result for /predict_proba')

def write_metrics(metrics):
    st.write('## Метрики качества (в процессе обучения бейзлайна):')
    df = pd.DataFrame(metrics, index=[0]).T
    df.columns = ['Значение метрики']
    st.write(df)
    logger.debug('Write result for /metrics')

def write_participants(participants):
    st.write('## Участники проекта:')
    st.write(participants)
    logger.debug('Write result for /participants')

def process_side_bar_inputs():
    st.sidebar.header('Данные о пациенте (пациентах)')
    user_input_df = sidebar_input_features()
    data = [row.to_dict() for _, row in user_input_df.iterrows()]
    st.write("## Ваши данные")
    st.table(user_input_df)
    return data
    

def sidebar_input_features():
    '''
    Рендеринг формы для загрузки csv файла или выбора параметров вручную
    '''

    uploaded_file = st.sidebar.file_uploader(label='Загрузить CSV  файл', type=['csv'])

    st.sidebar.markdown('**Или выставите параметры вручную**')

    # собираем лайтовые фичи
    age = st.sidebar.number_input("Возраст", 0, 120, 55)
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский", "Не могу сказать точно"))
    height = st.sidebar.number_input("Рост", 50, 250, 170)
    weight = st.sidebar.number_input("Вес", 10, 200, 75)
    ap_hi = st.sidebar.number_input("Верхнее давление", 80, 200, 120)
    ap_lo = st.sidebar.number_input("Нижнее давление", 50, 150, 80)
    cholesterol = st.sidebar.slider("Уровень холестерина", min_value=1, max_value=3, value=1, step=1)
    gluc =  st.sidebar.slider("Уровень глюкозы", min_value=1, max_value=3, value=1, step=1)
    smoke = st.sidebar.selectbox("Курите", ("Нет", "Да", "Не могу сказать точно"))
    alco = st.sidebar.selectbox("Употребляете алкоголь", ("Нет", "Да", "Не могу сказать точно"))
    active = st.sidebar.selectbox("Занимаетесь физической активностью", ("Нет", "Да", "Не могу сказать точно"))

    chest_pain_type = st.sidebar.selectbox("Тип боли в груди", ('Не могу сказать точно', 'Бессимптомный', 'Типичная стенокардия', 'Атипичная стенокардия', 'Неангинозная боль'))
    resting_blood_pressure = st.sidebar.number_input('Артериальное давление в состоянии покоя', 80, 200, 130)
    serum_cholestoral = st.sidebar.number_input("Холестерин в мг/дл)", 100, 300, 240)
    fasting_blood_sugar = st.sidebar.selectbox("Уровень сахара в крови натощак меньше 120 mg/d", ("Не могу сказать точно", "Нет", "Да"))
    resting_electrocardiographic = st.sidebar.selectbox("Результаты электрокардиографии в состоянии покоя", ("Не могу сказать точно", "Норма", "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)", "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса"))
    maximum_heart_rate_achieved = st.sidebar.number_input("Максимальная достигнутая частота сердечных сокращений", 70, 200, 150)
    exercise_induced_angina = st.sidebar.selectbox("Имеется стенокардия, вызванная физической нагрузкой", ("Не могу сказать точно", "Нет", "Да"))
    oldpeak = st.sidebar.slider("Снижение ST сегмента электрокардиограммы вызванный физической нагрузкой", min_value=0, max_value=6, value=1, step=1)
    slope_peak_ST = st.sidebar.selectbox("Наклон сегмента ST при пиковой нагрузке", ("Не могу сказать точно", "Плоский", "Восходящий", "Нисходящий"))
    vessels_colored_flourosopy = st.sidebar.slider("Количество крупных сосудов", min_value=0, max_value=4, value=0, step=1)
    thallium_stress_test = st.sidebar.selectbox("Таллиевый стресс-тест", ("Не могу сказать точно", "Норма", "Ошибка", "Фиксированный дефект", "Обратимый дефект"))    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logger.debug('Uplodead CSV file')
    else:
        df = pd.DataFrame({
            # фичи Вани
            'Age': age,
            'Sex': sex,
            'CheastPainType': chest_pain_type,
            'RestingBP': resting_blood_pressure,
            'Cholesterol': serum_cholestoral,
            'FastingBS': fasting_blood_sugar,
            'RestingECG': resting_electrocardiographic,
            'MaxHR': maximum_heart_rate_achieved,
            'ExerciseAngina': exercise_induced_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': slope_peak_ST,
            'NumMajorVessels': vessels_colored_flourosopy,
            'Thal': thallium_stress_test,

            # фичи Дани
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }, index=[0])   
        logger.debug('Changed input forms')
    return df


if __name__ == "__main__":
    process_main_page()
