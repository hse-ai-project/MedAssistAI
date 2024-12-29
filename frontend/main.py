import logging
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from logging.handlers import TimedRotatingFileHandler
from json import loads

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
        # 👨🏼‍⚕️ Предсказание наличия сердечных заболевания
        🔬 Определяем, вероятность сердечных заболеваний на основе информации о пациенте (пациентах)
        """
    )

def process_main_page():
    '''
    Рендеринг основных элементов страницы (заголовки, изображения, sidebar, кнопки)
    '''
    image = Image.open('data/ded.jpg')
    st.image(image)
    patients = process_side_bar_inputs()
    fit_expander = st.expander("Fit")
    set_model_expander = st.expander("Set model")
    predict_expander = st.expander("Predict")
    predict_proba_expander = st.expander("Predict proba")
    models_expander = st.expander("Models")
    update_model_expander = st.expander("Update model")
    participants_expander = st.expander("Participants")

    with fit_expander:
        logger.debug('\'Fit\' button clicked')
        st.write("## Обучить модель с заданными гиперпараметрами на ваших данных")
        model_id = st.text_input('Введите ID новой модели', placeholder='best_model_in_the_world_1')
        train_data = st.file_uploader(label='Загрузить CSV файл c обучающими данными', type=['csv'])
        params = st.file_uploader(label='Загрузить JSON файл c гиперпараметрами', type=['json'])
        timeout = st.number_input('Введите время для таймаута (сек)', min_value=1, max_value=10000, value=10)
        send_fit_request = st.button("Отправить запрос", key='send_fit_request')

        if send_fit_request:
            red_flag = False  # если не хватает данных, то не даст отправить запрос к серверу
            if model_id is None or model_id == '':
                st.write('#### Не заполнен ID модели!')
                red_flag = True
            if train_data is None:
                st.write('#### Не загружены обучающие данные (CSV файл)!')
                red_flag = True
            if params is None:
                st.write('#### Не загружены гиперпараметры (JSON файл)!')
                red_flag = True
            if timeout is None:
                st.write('#### Не установлено время таймаута!')
                red_flag = True
            
            if not red_flag:
                st.write("Запрос отправлен!")
                train_df = pd.read_csv(train_data)
                X = train_df.drop(columns=['target'])
                features = {}
                for col in X.columns:
                    features[col] = X[col].to_list()
                target = train_df['target'].to_list()
                fit_json_data = {
                    'id': model_id,
                    'hyperparameters': loads(params.read()),
                    'train_data': {
                        'features': features, 
                        'target': target
                    },
                    'timeout': 10
                }
                try:
                    fitting_result = requests.post("http://127.0.0.1:8000/model/fit", json=fit_json_data).json()
                    write_fit_result(fitting_result)
                except Exception as err:
                    logger.error("Cant get response from API for /fit: " + str(err))
                    st.write('#### Не удалось выполнить запрос к API: ')
                    st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку
            

    with set_model_expander:
        logger.debug('\'Set model\' button clicked')
        st.write("## Установите модель в качестве активной для дальнейших предсказаний")
        model_id = st.text_input('Введите ID обученной модели', placeholder='best_model_in_the_world_1')
        send_set_model_request = st.button("Отправить запрос", key='send_set_model_request')
        if send_set_model_request:
            set_json_data = {
                'model_id': model_id
            }
            st.write("Запрос отправлен!")
            try:
                set_model_result = requests.post("http://127.0.0.1:8000/model/set_model", json=set_json_data).json()
                if 'detail' in set_model_result:  # поднялся HTTPException
                    st.write(f'#### {set_model_result['detail']}')
                else:
                    st.write(set_model_result)
            except Exception as err:
                logger.error("Cant get response from API for /set_model: " + str(err))
                st.write('#### Не удалось выполнить запрос к API: ')
                st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку
    

    with update_model_expander:
        logger.debug('\'Update model\' button clicked')
        st.write("## Обновление существующей модели новыми данными")
        model_id = st.text_input('Введите ID модели для обновления', placeholder='best_model_in_the_world_1')
        train_data = st.file_uploader(label='Загрузить CSV файл c новыми тренировочными данными', type=['csv'])
        send_update_model_request = st.button("Отправить запрос", key='send_update_model_request')
       

        if send_update_model_request: 
            red_flag = False
            if model_id is None or model_id == '':
                st.write('#### Не заполнен ID модели!')
                red_flag = True
            if train_data is None:
                st.write('#### Не загружены обучающие данные (CSV файл)!')
                red_flag = True

            if not red_flag:
                train_df = pd.read_csv(train_data)
                X = train_df.drop(columns=['target'])
                features = {}
                for col in X.columns:
                    features[col] = X[col].to_list()
                target = train_df['target'].to_list()
                update_json_data = {
                    'model_id': model_id,
                    'features': features,
                    'target': target
                }
                st.write("Запрос отправлен!")
                try:
                    update_model_result = requests.post(f"http://127.0.0.1:8000/model/update_model/{model_id}", json=update_json_data).json()
                    if 'detail' in update_model_result:
                        st.write(f'#### {update_model_result['detail']}')
                    else:
                        st.write(update_model_result)
                except Exception as err:
                    logger.error("Cant get response from API for /update_model/{model_id}: " + str(err))
                    st.write('#### Не удалось выполнить запрос к API: ')
                    st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку

    with predict_expander:
        logger.debug('\'Predict\' button clicked')
        st.write("## Предсказание на основе данных пациентов")
        st.write("Требуется заполнить sidebar: загрузить CSV файл или выставить параметры вручную")
        send_predict_request = st.button("Отправить запрос", key='send_predict_request')
        if send_predict_request:
            predict_json_data = {
                'patients': patients
            }
            st.write("Запрос отправлен!")
            try:
                predict_model_result = requests.post("http://127.0.0.1:8000/model/predict", json=predict_json_data).json()
                if 'detail' in predict_model_result:
                    st.write(f'#### {predict_model_result['detail']}')
                else:
                    write_prediction(predict_model_result)
            except Exception as err:
                logger.error("Cant get response from API for /predict: " + str(err))
                st.write('#### Не удалось выполнить запрос к API: ')
                st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку

    with predict_proba_expander:
        logger.debug('\'Predict proba\' button clicked')
        st.write("## Вероятность сердечного заболевания на основе данных пациентов")
        st.write("Требуется заполнить sidebar: загрузить CSV файл или выставить параметры вручную")
        send_predict_proba_request = st.button("Отправить запрос", key='send_predict_proba_request')
        if send_predict_proba_request:
            predict_json_data = {
                'patients': patients
            }
            st.write("Запрос отправлен!")
            try:
                predict_proba_model_result = requests.post("http://127.0.0.1:8000/model/predict", json=predict_json_data).json()
                if 'detail' in predict_proba_model_result:
                    st.write(f'#### {predict_proba_model_result['detail']}')
                else:
                    probs = [row['probability'] for row in predict_proba_model_result]
                    write_prediction_proba(probs)
            except Exception as err:
                logger.error("Cant get response from API for /predict: " + str(err))
                st.write('#### Не удалось выполнить запрос к API: ')
                st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку


    with models_expander:
        logger.debug('\'Models\' button clicked')
        st.write("##  Получение списка всех доступных моделей и информации о них")
        send_models_request = st.button("Отправить запрос", key='send_models_request')
        if send_models_request:
            st.write("Запрос отправлен!")
            try:
                models_result = requests.get("http://127.0.0.1:8000/model/models").json()
                write_models(models_result)
            except Exception as err:
                logger.error("Cant get response from API for /models: " + str(err))
                st.write('#### Не удалось выполнить запрос к API: ')
                st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку

    
    with participants_expander:
        logger.debug('\'Participants\' button clicked')
        st.write("## Информация об участниках проекта")
        send_participants_request = st.button("Отправить запрос", key='send_participants_request')
        if send_participants_request:
            st.write("Запрос отправлен!")
            try:
                participants_result = requests.get("http://127.0.0.1:8000/participants").json()['status']
                write_participants(participants_result)
            except Exception as err:
                logger.error("Cant get response from API for /participants: " + str(err))
                st.write('#### Не удалось выполнить запрос к API: ')
                st.write(str(err))  # если не хотим выводить ошибку на клиент - закомментить эту строчку


# Ниже представлены функции для отрисовки полученных данных от FastAPI
def write_fit_result(fit_result):
    st.write("## Результат обучения")
    st.table(pd.DataFrame(fit_result))

def write_models(models):
    st.write("## Доступные модели")
    st.table(pd.DataFrame(models))

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
    st.sidebar.header('🪪 Данные о пациенте (пациентах) для постановки диагноза')
    user_input_df = sidebar_input_features()
    data = [row.to_dict() for _, row in user_input_df.iterrows()]
    st.write("## 🩺 Данные пациентов")
    st.table(user_input_df)
    return data
    

def sidebar_input_features():
    '''
    Рендеринг формы для загрузки csv файла или выбора параметров вручную
    '''

    uploaded_file = st.sidebar.file_uploader(label='Загрузить CSV  файл', type=['csv'])

    st.sidebar.markdown('**Или выставите параметры вручную**')
    age = st.sidebar.number_input("Возраст", 0, 120, 55)
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
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
    thallium_stress_test = st.sidebar.selectbox("Таллиевый стресс-тест", ("Не могу сказать точно", "Норма", "Фиксированный дефект", "Обратимый дефект"))    

    translation = {
        'Мужской': 1,
        'Женский': 0,
        'Не могу сказать точно': None,
        'Нет': 0,
        'Да': 1,
        'Бессимптомный': 3,
        'Типичная стенокардия': 0,
        'Атипичная стенокардия': 1,
        'Неангинозная боль': 2,
        'Норма': 0,
        "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)": 1,
        "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса": 2,
        'Плоский': 1,
        'Восходящий': 0,
        'Нисходящий': 2,
        'Фиксированный дефект': 1,
        'Обратимый дефект': 2
    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logger.debug('Uplodead CSV file')
    else:
        df = pd.DataFrame({
            # фичи Вани
            'age': age,
            'sex': translation[sex],
            'chest_pain_type': translation[chest_pain_type],
            'resting_bp': resting_blood_pressure,
            'cholesterol': serum_cholestoral,
            'fasting_bs': translation[fasting_blood_sugar],
            'resting_ecg': translation[resting_electrocardiographic],
            'max_hr': maximum_heart_rate_achieved,
            'exercise_angina': translation[exercise_induced_angina],
            'oldpeak': oldpeak,
            'st_slope': translation[slope_peak_ST],
            'num_major_vessels': vessels_colored_flourosopy,
            'thal': translation[thallium_stress_test],

            # фичи Дани
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': translation[smoke],
            'alco': translation[alco],
            'active': translation[active]
        }, index=[0])   
        logger.debug('Changed input forms')
    return df


if __name__ == "__main__":
    process_main_page()
