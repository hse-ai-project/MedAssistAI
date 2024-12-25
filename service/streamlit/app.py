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
    '''
    Рендеринг основных элементов страницы (заголовки, изображения, sidebar, кнопки)
    '''
    image = Image.open('data/ded.jpg')
    st.image(image)
    data = process_side_bar_inputs()
    if st.button("Предсказать"):
        # ручка /predict
        try:
            prediction = requests.post("http://localhost:8000/predict", json=data)
        except Exception as err:
            prediction = 'Не удалось выполнить запрос к API'

        if prediction == 1:
            write_prediction("У вас выявлено возможное сердечное заболевание! Срочно обратитесь к врачу!")
        elif prediction == 0:
            write_prediction("У вас не выявлено сердечных заболеваний!")
        else:
            write_prediction(prediction)

    if st.button("Предсказать вероятности"):
        # ручка /predict_proba
        try:
            prediction_probs = requests.post("http://localhost:8000/predict_proba", json=data)
        except Exception as err:
            prediction_probs = 'Не удалось выполнить запрос к API'
        write_prediction_proba(prediction_probs)

    if st.button("Метрики качества модели"):
        # ручка /metrics
        try:
            metrics = requests.get('http://localhost:8000/metrics')
        except Exception as err:
            metrics = ['Не удалось выполнить запрос к API']
        write_metrics(metrics)

    if st.button("Участники проекта"):
        # ручка /participants
        try:
            participants = requests.get('http://localhost:8000/participants')
        except:
            participants = 'Не удалось выполнить запрос к API'
        write_participants(participants)


# Ниже представлены функции для отрисовки полученных данных от FastAPI
def write_prediction(prediction):
    st.write("## Поставленный диагноз")
    st.write(prediction)

def write_prediction_proba(prediction_probs):
    st.write("## Вероятность поставленного диагноза")
    st.write(pd.DataFrame(prediction_probs, columns=['Вероятность'], index=['Нет заболевания', 'Есть заболевание']))

def write_metrics(metrics):
    st.write('## Метрики качества (в процессе обучения бейзлайна):')
    df = pd.DataFrame(metrics, index=[0]).T
    df.columns = ['Значение метрики']
    st.write(df)

def write_participants(participants):
    st.write('## Участники проекта:')
    st.write(participants)

def process_side_bar_inputs():
    st.sidebar.header('Данные о пациенте (пациентах)')
    user_input_df = sidebar_input_features()
    data = [row.to_json() for _, row in user_input_df.iterrows()]
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
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    height = st.sidebar.number_input("Рост", 50, 250, 170)
    weight = st.sidebar.number_input("Вес", 10, 200, 75)
    ap_hi = st.sidebar.number_input("Верхнее давление", 80, 200, 120)
    ap_lo = st.sidebar.number_input("Нижнее давление", 50, 150, 80)
    cholesterol = st.sidebar.slider("Уровень холестерина", min_value=1, max_value=3, value=1, step=1)
    gluc =  st.sidebar.slider("Уровень глюкозы", min_value=1, max_value=3, value=1, step=1)
    smoke = st.sidebar.selectbox("Курите", ("Нет", "Да"))
    alco = st.sidebar.selectbox("Употребляете алкоголь", ("Нет", "Да"))
    active = st.sidebar.selectbox("Занимаетесь физической активностью", ("Нет", "Да"))

    chest_pain_type = st.sidebar.selectbox("Тип боли в груди", ('Бессимптомный', 'Типичная стенокардия', 'Атипичная стенокардия', 'Неангинозная боль'))
    resting_blood_pressure = st.sidebar.number_input("Артериальное давление в состоянии покоя", 80, 200, 130)
    serum_cholestoral = st.sidebar.number_input("Холестерин в мг/дл)", 100, 300, 240)
    fasting_blood_sugar = st.sidebar.selectbox("Уровень сахара в крови натощак меньше 120 mg/d", ("Нет", "Да"))
    resting_electrocardiographic = st.sidebar.selectbox("Результаты электрокардиографии в состоянии покоя", ("Норма", "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)", "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса"))
    maximum_heart_rate_achieved = st.sidebar.number_input("Максимальная достигнутая частота сердечных сокращений", 70, 200, 150)
    exercise_induced_angina = st.sidebar.selectbox("Имеется стенокардия, вызванная физической нагрузкой", ("Нет", "Да"))
    oldpeak = st.sidebar.slider("Снижение ST сегмента электрокардиограммы вызванный физической нагрузкой", min_value=0, max_value=6, value=1, step=1)
    slope_peak_ST = st.sidebar.selectbox("Наклон сегмента ST при пиковой нагрузке", ("Плоский", "Восходящий", "Нисходящий"))
    vessels_colored_flourosopy = st.sidebar.slider("Количество крупных сосудов", min_value=0, max_value=4, value=0, step=1)
    thallium_stress_test = st.sidebar.selectbox("Таллиевый стресс-тест", ("Нормальное значение", "Ошибка", "Фиксированный дефект", "Обратимый дефект"))    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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

    return df


if __name__ == "__main__":
    process_main_page()
