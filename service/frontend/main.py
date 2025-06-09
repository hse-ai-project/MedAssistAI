"""
Streamlit приложение
"""

import logging
from json import loads
from logging.handlers import TimedRotatingFileHandler
import requests
import pandas as pd
import streamlit as st
from PIL import Image


class ServerException(Exception):
    pass


def setup_logger():
    """Конфигурация логгера для последующей его использования"""
    logger_ = logging.getLogger("frontend")
    if not logger_.hasHandlers():
        logger_.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(
            "logs/logs.log", when="D", interval=1)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger_.addHandler(handler)
    return logger_


logger = setup_logger()

st.write(
    """
        # 👨🏼‍⚕️ Предсказание наличия сердечных заболевания
        🔬 Определяем, вероятность сердечных заболеваний на основе информации о пациенте (пациентах)
        """
)


def process_main_page():
    """
    Рендеринг основных элементов страницы
    (заголовки, изображения, sidebar, кнопки)
    """
    image = Image.open("data/ded.jpg")
    st.image(image)
    patients = process_side_bar_inputs()
    prompt_expander = st.expander("User Prompt")
    fit_expander = st.expander("Fit")
    set_model_expander = st.expander("Set model")
    predict_expander = st.expander("Predict")
    predict_proba_expander = st.expander("Predict proba")
    models_expander = st.expander("Models")
    update_model_expander = st.expander("Update model")
    participants_expander = st.expander("Participants")

    with prompt_expander:
        st.write(
            "## Поиск симптомов по описанию")
        user_prompt_text = st.text_input(
            "Укажите все свои симптомы и физические параметры",
            placeholder="Возраст 52 года, температура 38.3, кружится голова, давление 120/70."
        )
        send_user_prompt_request = st.button(
            "Отправить запрос", key="send_user_prompt_request"
        )
        if send_user_prompt_request:
            logger.debug("'Prompt' button clicked")
            prompt_json_data = {"text": user_prompt_text}
            st.write("Запрос отправлен!")
            try:
                logger.debug("'Prompt' request has been sent")
                user_prompt_result = requests.post(
                    "http://fastapi:8000/model/predict_from_text",
                    json=prompt_json_data,
                    timeout=10,
                ).json()
                logger.debug("Got response from API for /predict_from_text")
                if "prediction" in user_prompt_result:
                    pred = [
                        (
                            "У вас обнаружено возможное сердечное \
                            заболевание! Срочно обратитесь к врачу!"
                            if user_prompt_result["prediction"]["prediction"] == 1
                            else "У вас не выявлено сердечных заболеваний"
                        )
                    ]
                    pred_proba = [
                        str(round(user_prompt_result["prediction"]["probability"], 4))
                    ]
                    logger.debug("Prediction result: %s", str(pred))
                    logger.debug("Prediction proba result: %s", str(pred_proba))
                    write_prediction(pred)
                    st.table(
                        pd.DataFrame(
                            pred_proba,
                            columns=["Вероятность сердечного заболевания"])
                    )
                else:
                    if "message" in user_prompt_result:
                        st.write(user_prompt_result['message'])
                    else:
                        st.write(user_prompt_result)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /predict_from_text: %s",
                    str(err))
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку

    with fit_expander:
        st.write("## Обучить модель с заданными \
                 гиперпараметрами на ваших данных")
        model_id = st.text_input(
            "Введите ID новой модели", placeholder="best_model_in_the_world_1"
        )
        train_data = st.file_uploader(
            label="Загрузить CSV файл c обучающими данными", type=["csv"]
        )
        params = st.file_uploader(
            label="Загрузить JSON файл c гиперпараметрами", type=["json"]
        )
        timeout = st.number_input(
            "Введите время для таймаута (сек)",
            min_value=1,
            max_value=10000,
            value=10
        )
        send_fit_request = st.button(
            "Отправить запрос", key="send_fit_request")

        if send_fit_request:
            logger.debug("'Fit' button clicked")
            red_flag = (
                False
                # если не хватает данных, то не даст отправить запрос к серверу
            )
            if model_id is None or model_id == "":
                logger.error("No model_id in 'Fit'!")
                st.write("#### Не заполнен ID модели!")
                red_flag = True
            if train_data is None:
                logger.error("No CSV-file in 'Fit'!")
                st.write("#### Не загружены обучающие данные (CSV файл)!")
                red_flag = True
            if params is None:
                logger.error("No hyperparameters in 'Fit'!")
                st.write("#### Не загружены гиперпараметры (JSON файл)!")
                red_flag = True
            if timeout is None:
                logger.error("No timeout in 'Fit'!")
                st.write("#### Не установлено время таймаута!")
                red_flag = True

            if not red_flag:
                st.write("Запрос отправлен!")
                train_df = pd.read_csv(train_data)
                x = train_df.drop(columns=["target"])
                features = {}
                for col in x.columns:
                    features[col] = x[col].to_list()
                target = train_df["target"].to_list()
                fit_json_data = {
                    "id": model_id,
                    "hyperparameters": loads(params.read()),
                    "train_data": {"features": features, "target": target},
                    "timeout": 10,
                }
                try:
                    logger.debug("'Fit' request has been sent")
                    fitting_result = requests.post(
                        "http://fastapi:8000/model/fit",
                        json=fit_json_data,
                        timeout=10
                    ).json()
                    logger.debug("Got response from API for /fit")
                    if "message" in fitting_result:
                        logger.debug(
                            "Fitting result: %s", str(
                                fitting_result["message"])
                        )
                        st.write(f"#### {fitting_result['message']}")
                    else:
                        logger.debug("Fitting result: %s", str(fitting_result))
                        st.write(fitting_result)
                except ServerException as err:
                    logger.error(
                        "Cant get response from API for /fit: %s", str(err))
                    st.write("#### Не удалось выполнить запрос к API: ")
                    st.write(
                        str(err)
                    )  # если не хотим выводить ошибку на клиент -
                    # закомментить эту строчку

    with set_model_expander:
        st.write(
            "## Установите модель в качестве активной \
            для дальнейших предсказаний")
        model_id = st.text_input(
            "Введите ID обученной модели",
            placeholder="best_model_in_the_world_1"
        )
        send_set_model_request = st.button(
            "Отправить запрос", key="send_set_model_request"
        )
        if send_set_model_request:
            logger.debug("'Set model' button clicked")
            set_json_data = {"model_id": model_id}
            st.write("Запрос отправлен!")
            try:
                logger.debug("'Set model' request has been sent")
                set_model_result = requests.post(
                    "http://fastapi:8000/model/set_model",
                    json=set_json_data,
                    timeout=10,
                ).json()
                logger.debug("Got response from API for /set_model")
                if "detail" in set_model_result:  # поднялся HTTPException
                    logger.debug(set_model_result["detail"])
                    st.write(f"#### {set_model_result['detail']}")
                elif "status" in set_model_result:
                    if set_model_result["status"] == "success":
                        logger.debug("Success setting model as active")
                        st.write(
                            f"#### Модель {model_id} теперь является активной")
                    else:
                        logger.debug(
                            "Setting model: %s", str(
                                set_model_result["status"])
                        )
                        st.write(
                            f"#### Статус установки модели \
                            в качестве активной: {set_model_result['status']}"
                        )
                else:
                    logger.debug("Setting model: %s", str(set_model_result))
                    st.write(set_model_result)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /set_model: %s",
                    str(err))
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку

    with update_model_expander:
        st.write("## Обновление существующей модели новыми данными")
        model_id = st.text_input(
            "Введите ID модели для обновления",
            placeholder="best_model_in_the_world_1"
        )
        train_data = st.file_uploader(
            label="Загрузить CSV файл c новыми тренировочными данными",
            type=["csv"]
        )
        send_update_model_request = st.button(
            "Отправить запрос", key="send_update_model_request"
        )

        if send_update_model_request:
            logger.debug("'Update model' button clicked")
            red_flag = False
            if model_id is None or model_id == "":
                st.write("#### Не заполнен ID модели!")
                logger.error("No model_id in 'Update model'!")
                red_flag = True
            if train_data is None:
                st.write("#### Не загружены обучающие данные (CSV файл)!")
                logger.error("No CSV-file in 'Update model'!")
                red_flag = True

            if not red_flag:
                train_df = pd.read_csv(train_data)
                x = train_df.drop(columns=["target"])
                features = {}
                for col in x.columns:
                    features[col] = x[col].to_list()
                target = train_df["target"].to_list()
                update_json_data = {
                    "model_id": model_id,
                    "features": features,
                    "target": target,
                }
                st.write("Запрос отправлен!")
                try:
                    logger.debug("'Update model' request has been sent")
                    update_model_result = requests.post(
                        f"http://fastapi:8000/model/update_model/{model_id}",
                        json=update_json_data,
                        timeout=10,
                    ).json()
                    logger.debug(
                        "Got response from API for /update_model/%s", model_id)
                    if "message" in update_model_result:
                        logger.debug(
                            "Updating model: %s", str(
                                update_model_result["message"])
                        )
                        st.write(f"#### {update_model_result['message']}")
                    elif "detail" in update_model_result:
                        logger.debug(
                            "Updating model: %s", str(
                                update_model_result["detail"])
                        )
                        st.write(f"#### {update_model_result['detail']}")
                    else:
                        logger.debug(
                            "Updating model: %s",
                            str(update_model_result))
                        st.write(update_model_result)
                except ServerException as err:
                    logger.error(
                        "Cant get response from API for \
                        /update_model/{model_id}: %s",
                        str(err),
                    )
                    st.write("#### Не удалось выполнить запрос к API: ")
                    st.write(
                        str(err)
                    )  # если не хотим выводить ошибку на клиент -
                    # закомментить эту строчку

    with predict_expander:
        st.write("## Предсказание на основе данных пациентов")
        st.write(
            "Требуется заполнить sidebar: загрузить CSV файл \
            или выставить параметры вручную"
        )
        send_predict_request = st.button(
            "Отправить запрос", key="send_predict_request")
        if send_predict_request:
            logger.debug("'Predict' button clicked")
            predict_json_data = {"patients": patients}
            st.write("Запрос отправлен!")
            try:
                logger.debug("'Predict' request has been sent")
                predict_model_result = requests.post(
                    "http://fastapi:8000/model/predict",
                    json=predict_json_data,
                    timeout=10,
                ).json()
                logger.debug("Got response from API for /predict")
                if "predictions" in predict_model_result:
                    pred = [
                        (
                            "У вас обнаружено возможное сердечное \
                            заболевание! Срочно обратитесь к врачу!"
                            if value["prediction"] == 1
                            else "У вас не выявлено сердечных заболеваний"
                        )
                        for value in predict_model_result["predictions"]
                    ]
                    logger.debug("Prediction result: %s", str(pred))
                    write_prediction(pred)
                elif "detail" in predict_model_result:
                    logger.debug(
                        "Prediction result: %s", str(
                            predict_model_result["detail"])
                    )
                    st.write(f"#### {predict_model_result['detail']}")
                else:
                    logger.debug(
                        "Prediction result: %s",
                        str(predict_model_result))
                    write_prediction(predict_model_result)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /predict: %s",
                    str(err))
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку

    with predict_proba_expander:
        st.write("## Вероятность сердечного заболевания \
                 на основе данных пациентов")
        st.write(
            "Требуется заполнить sidebar: загрузить CSV файл \
            или выставить параметры вручную"
        )
        send_predict_proba_request = st.button(
            "Отправить запрос", key="send_predict_proba_request"
        )
        if send_predict_proba_request:
            logger.debug("'Predict proba' button clicked")
            predict_json_data = {"patients": patients}
            st.write("Запрос отправлен!")
            try:
                logger.debug(
                    "'Predict' (for predict probability) request has been sent"
                )
                predict_proba_model_result = requests.post(
                    "http://fastapi:8000/model/predict",
                    json=predict_json_data,
                    timeout=10,
                ).json()
                logger.debug("Got response from API for /predict")
                if "predictions" in predict_proba_model_result:
                    pred_proba = [
                        value["probability"]
                        for value in predict_proba_model_result["predictions"]
                    ]
                    logger.debug("Prediction result: %s", str(pred_proba))
                    write_prediction_proba(pred_proba)
                elif "detail" in predict_proba_model_result:
                    logger.debug(
                        "Prediction result: %s",
                        str(predict_proba_model_result["detail"]),
                    )
                    st.write(f"#### {predict_proba_model_result['detail']}")
                else:
                    probs = [row["probability"]
                             for row in predict_proba_model_result]
                    logger.debug("Prediction result: %s", str(probs))
                    write_prediction_proba(probs)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /predict: %s",
                    str(err))
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку

    with models_expander:
        st.write("##  Получение списка всех доступных моделей \
                 и информации о них")
        send_models_request = st.button(
            "Отправить запрос", key="send_models_request")
        if send_models_request:
            logger.debug("'Models' button clicked")
            st.write("Запрос отправлен!")
            try:
                logger.debug("'Models' request has been sent")
                models_result = requests.get(
                    "http://fastapi:8000/model/models", timeout=10
                ).json()
                logger.debug("Got response from API for /models")
                if "models" in models_result:
                    write_models(models_result["models"])
                else:
                    write_models(models_result)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /models: %s",
                    str(err))
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку

    with participants_expander:
        st.write("## Информация об участниках проекта")
        send_participants_request = st.button(
            "Отправить запрос", key="send_participants_request"
        )
        if send_participants_request:
            logger.debug("'Participants' button clicked")
            st.write("Запрос отправлен!")
            try:
                logger.debug("'Participants' request has been sent")
                participants_result = requests.get(
                    "http://fastapi:8000/participants", timeout=10
                ).json()["status"]
                logger.debug("Got response from API for /participants")
                write_participants(participants_result)
            except ServerException as err:
                logger.error(
                    "Cant get response from API for /participants: %s", str(
                        err)
                )
                st.write("#### Не удалось выполнить запрос к API: ")
                st.write(
                    str(err)
                )  # если не хотим выводить ошибку на клиент -
                # закомментить эту строчку


# Ниже представлены функции для отрисовки полученных данных от fastapi
def write_models(models):
    """Отображение доступных моделей"""
    st.write("## Доступные модели")
    st.table(models)


def write_prediction(prediction):
    """Отображение полученных прогнозов"""
    st.write("## Поставленный диагноз")
    st.table(pd.DataFrame(prediction, columns=["Диагноз"]))


def write_prediction_proba(prediction_probs):
    """Отображение вероятностей полученных прогнозов"""
    st.write("## Вероятность поставленного диагноза")
    st.table(
        pd.DataFrame(
            prediction_probs,
            columns=["Вероятность сердечного заболевания"])
    )


def write_participants(participants):
    """Отображение участников проекта"""
    st.write("## Участники проекта:")
    st.write(participants)


def process_side_bar_inputs():
    """Загрузка данных из сайдбара о пациентах"""
    st.sidebar.header(
        "🪪 Данные о пациенте (пациентах) для постановки диагноза")
    user_input_df = sidebar_input_features()
    data = [row.to_dict() for _, row in user_input_df.iterrows()]
    st.write("## 🩺 Данные пациентов")
    st.table(user_input_df)
    return data


def sidebar_input_features():
    """
    Рендеринг формы для загрузки данных через csv файл
    или через выбор параметров вручную
    """

    uploaded_file = st.sidebar.file_uploader(
        label="Загрузить CSV  файл", type=["csv"])

    st.sidebar.markdown("**Или выставите параметры вручную**")
    age = st.sidebar.number_input("Возраст", 0, 120, 55)
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    height = st.sidebar.number_input("Рост", 50, 250, 170)
    weight = st.sidebar.number_input("Вес", 10, 200, 75)
    ap_hi = st.sidebar.number_input("Верхнее давление", 80, 200, 120)
    ap_lo = st.sidebar.number_input("Нижнее давление", 50, 150, 80)
    cholesterol = st.sidebar.slider(
        "Уровень холестерина", min_value=1, max_value=3, value=1, step=1
    )
    gluc = st.sidebar.slider(
        "Уровень глюкозы", min_value=1, max_value=3, value=1, step=1
    )
    smoke = st.sidebar.selectbox(
        "Курите", ("Нет", "Да", "Не могу сказать точно"))
    alco = st.sidebar.selectbox(
        "Употребляете алкоголь", ("Нет", "Да", "Не могу сказать точно")
    )
    active = st.sidebar.selectbox(
        "Занимаетесь физической активностью", ("Нет",
                                               "Да", "Не могу сказать точно")
    )

    chest_pain_type = st.sidebar.selectbox(
        "Тип боли в груди",
        (
            "Не могу сказать точно",
            "Бессимптомный",
            "Типичная стенокардия",
            "Атипичная стенокардия",
            "Неангинозная боль",
        ),
    )
    resting_blood_pressure = st.sidebar.number_input(
        "Артериальное давление в состоянии покоя", 80, 200, 130
    )
    serum_cholestoral = st.sidebar.number_input(
        "Холестерин в мг/дл)", 100, 300, 240)
    fasting_blood_sugar = st.sidebar.selectbox(
        "Уровень сахара в крови натощак меньше 120 mg/d",
        ("Не могу сказать точно", "Нет", "Да"),
    )
    resting_electrocardiographic = st.sidebar.selectbox(
        "Результаты электрокардиографии в состоянии покоя",
        (
            "Не могу сказать точно",
            "Норма",
            "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)",
            "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса",
        ),
    )
    maximum_heart_rate_achieved = st.sidebar.number_input(
        "Максимальная достигнутая частота сердечных сокращений", 70, 200, 150
    )
    exercise_induced_angina = st.sidebar.selectbox(
        "Имеется стенокардия, вызванная физической нагрузкой",
        ("Не могу сказать точно", "Нет", "Да"),
    )
    oldpeak = st.sidebar.slider(
        "Снижение ST сегмента электрокардиограммы вызванный физической нагрузкой",
        min_value=0,
        max_value=6,
        value=1,
        step=1,
    )
    slope_peak_st = st.sidebar.selectbox(
        "Наклон сегмента ST при пиковой нагрузке",
        ("Не могу сказать точно", "Плоский", "Восходящий", "Нисходящий"),
    )
    vessels_colored_flourosopy = st.sidebar.slider(
        "Количество крупных сосудов", min_value=0, max_value=4, value=0, step=1
    )
    thallium_stress_test = st.sidebar.selectbox(
        "Таллиевый стресс-тест",
        ("Не могу сказать точно", "Норма",
         "Фиксированный дефект", "Обратимый дефект"),
    )

    translation = {
        "Мужской": 1,
        "Женский": 0,
        "Не могу сказать точно": None,
        "Нет": 0,
        "Да": 1,
        "Бессимптомный": 3,
        "Типичная стенокардия": 0,
        "Атипичная стенокардия": 1,
        "Неангинозная боль": 2,
        "Норма": 0,
        "Наличие аномалии зубца ST-T (инверсия зубца T и/или подъем или снижение ST > 0,05 мВ)": 1,
        "Демонстрация вероятной или определенной гипертрофии левого желудочка по критериям Эстеса": 2,
        "Плоский": 1,
        "Восходящий": 0,
        "Нисходящий": 2,
        "Фиксированный дефект": 1,
        "Обратимый дефект": 2,
    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logger.debug("Uplodead CSV file")
    else:
        df = pd.DataFrame(
            {
                # Модель 1
                "Age": age,
                "Sex": translation[sex],
                "CheastPainType": translation[chest_pain_type],
                "RestingBP": resting_blood_pressure,
                "Cholesterol": serum_cholestoral,
                "FastingBS": translation[fasting_blood_sugar],
                "RestingECG": translation[resting_electrocardiographic],
                "MaxHR": maximum_heart_rate_achieved,
                "ExerciseAngina": translation[exercise_induced_angina],
                "Oldpeak": oldpeak,
                "ST_Slope": translation[slope_peak_st],
                "NumMajorVessels": vessels_colored_flourosopy,
                "Thal": translation[thallium_stress_test],
                # Модель 2
                "age": age,
                "gender": translation[sex],
                "cholesterol": cholesterol,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "gluc": gluc,
                "smoke": translation[smoke],
                "alco": translation[alco],
                "active": translation[active],
            },
            index=[0],
        )
    return df


if __name__ == "__main__":
    process_main_page()
