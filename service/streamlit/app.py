import requests
import pandas as pd
import streamlit as st
from PIL import Image

st.write(
        """
        # Предсказание наличия сердечных заболевания
        Определяем, вероятность сердечных заболеваний на основе информации о пациенте (пациентах)
        """
    )

def process_main_page():
    image = Image.open('data/ded.jpg')
    st.image(image)
    data = process_side_bar_inputs()
    if st.button("/predict"):
        try:
            prediction = requests.post("http://localhost:8000/predict", json=data.to_json())
        except Exception as err:
            prediction = 'Не удалось выполнить запрос к API'
        write_prediction(prediction)

    if st.button("/predict_proba"):
        try:
            prediction_probs = requests.post("http://localhost:8000/predict_proba", json=data.to_json())
        except Exception as err:
            prediction_probs = 'Не удалось выполнить запрос к API'
        write_prediction_proba(prediction_probs)

    if st.button("/metrics"):
        try:
            metrics = requests.get('http://localhost:8000/metrics')
        except Exception as err:
            metrics = 'Не удалось выполнить запрос к API'
        write_metrics(metrics)

    if st.button("/participants"):
        try:
            participants = requests.get('http://localhost:8000/participants')
        except:
            participants = '''
                Васильев Даниил — @daniel_vasiliev, AristanD
                 
                Мартынов Александр — @martynovall, alexmart811
                 
                Черных Иван — @zzzippp, xfiniks
                 
                Ляпин Данила — @danila_lyapin, Lak1n26
                 
                Куратор:
                Малюшитский Кирилл — @malyushitsky, malyushitsky
                 '''
        write_participants(participants)


# Ниже представлены функции для отрисовки полученных данных от FastAPI
def write_prediction(prediction):
    st.write("## Поставленный диагноз")
    st.write(prediction)

def write_prediction_proba(prediction_probs):
    st.write("## Вероятность поставленного диагноза")
    st.write(prediction_probs)

def write_metrics(metrics):
    st.write('**Метрики качества (в процессе обучения бейзлайна):**')
    st.write(metrics)

def write_participants(participants):
    st.write('**Участники проекта (ФИ, github, tg):**')
    st.write(participants)

def process_side_bar_inputs():
    st.sidebar.header('Данные о пациенте (пациентах)')
    user_input_df = sidebar_input_features()
    st.write("## Ваши данные")
    st.write(user_input_df)
    return user_input_df
    

def sidebar_input_features():

    uploaded_file = st.sidebar.file_uploader(label='Загрузить CSV  файл', type=['csv'])

    st.sidebar.markdown('**Или выставите параметры вручную**')

    # собираем лайтовые фичи
    age = st.sidebar.number_input("Возраст", 0, 120, 55)
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    height = st.sidebar.number_input("Рост", 50, 250, 170)
    weight = st.sidebar.number_input("Вес", 10, 200, 75)
    ap_hi = st.sidebar.number_input("Верхнее давление", 80, 200, 120)
    ap_lo = st.sidebar.number_input("Нижнее давление", 50, 150, 80)
    cholesterol =  st.sidebar.number_input("Уровень холестерина", 1, 3, 1)
    gluc = st.sidebar.number_input("Уровень глюкозы", 1, 3, 1)
    smoke = st.sidebar.selectbox("Курите", ("Нет", "Да"))
    alco = st.sidebar.selectbox("Употребляете алкоголь", ("Нет", "Да"))
    active = st.sidebar.selectbox("Занимаетесь физической активностью", ("Нет", "Да"))

    
    # пошла жесть
    chest_pain_type = st.sidebar.selectbox("Тип боли в груди", ('Бессимптомный', 'Типичная стенокардия', 'Атипичная стенокардия', 'Неангинозная боль'))
    resting_blood_pressure = st.sidebar.number_input("Артериальное давление в состоянии покоя", 80, 200, 130)
    serum_cholestoral = st.sidebar.number_input("Холестерин в мг/дл)", 100, 300, 240)
    fasting_blood_sugar = st.sidebar.selectbox("Уровень сахара в крови натощак меньше 120 mg/d", ("Нет", "Да"))
    resting_electrocardiographic = st.sidebar.selectbox("Результаты электрокардиографии в состоянии покоя", ("Норма", "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)", "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса"))
    maximum_heart_rate_achieved = st.sidebar.number_input("Максимальная достигнутая частота сердечных сокращений", 70, 200, 150)
    exercise_induced_angina = st.sidebar.selectbox("Имеется стенокардия, вызванная физической нагрузкой", ("Нет", "Да"))
    oldpeak = st.sidebar.number_input("Снижение ST сегмента электрокардиограммы вызванный физической нагрузкой", 0, 6, 1)
    slope_peak_ST = st.sidebar.selectbox("Наклон сегмента ST при пиковой нагрузке", ("Плоский", "Восходящий", "Нисходящий"))
    vessels_colored_flourosopy = st.sidebar.number_input("Количество крупных сосудов , окрашенных с помощью флюороскопии", 0, 4, 0)
    thallium_stress_test = st.sidebar.selectbox("Таллиевый стресс-тест", ("Нормальное значение", "Ошибка", "Фиксированный дефект", "Обратимый дефект"))    

    # dependents = st.sidebar.slider("Иждивенцы: количество человек, зависящих от клиента", min_value=0, max_value=10, value=0, step=1)

    translation = {
        "Мужской": 1,
        "Женский": 0,
        "Да": 1,
        "Нет": 0,
        "Бессимптомный": 3,
        "Типичная стенокардия": 0,
        "Атипичная стенокардия": 1,
        "Неангинозная боль": 2,
        "Норма": 0,
        "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)": 1,
        "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса": 2,
        "Восходящий": 0,
        "Плоский": 1,
        "Нисходящий": 2,
        "Ошибка": 0,
        "Фиксированный дефект": 1,
        "Обратимый дефект": 3,
        "Нормальное значение": 2

    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        data = {
            # фичи Вани
            'Age': age,
            'Sex': translation[sex],
            'CheastPainType': translation[chest_pain_type],
            'RestingBP': resting_blood_pressure,
            'Cholesterol': serum_cholestoral,
            'FastingBS': translation[fasting_blood_sugar],
            'RestingECG': translation[resting_electrocardiographic],
            'MaxHR': maximum_heart_rate_achieved,
            'ExerciseAngina': translation[exercise_induced_angina],
            'Oldpeak': oldpeak,
            'ST_Slope': translation[slope_peak_ST],
            'NumMajorVessels': vessels_colored_flourosopy,
            'Thal': translation[thallium_stress_test],

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
        }
        df = pd.DataFrame(data, index=[0])
    return df


if __name__ == "__main__":
    process_main_page()
