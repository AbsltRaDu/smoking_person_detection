import streamlit as st
import requests
from PIL import Image, ImageDraw
import pandas as pd
import io

st.set_page_config(page_title="Cigarette Detection", layout="centered")
st.title("🚬 Детекция сигарет с помощью YOLO")


st.sidebar.header("Настройки")
API_URL = st.sidebar.text_input("Адрес API", "http://localhost:8000/predict")
conf_threshold = st.sidebar.slider(
    "Порог уверенности (Confidence)",
    min_value=0.0,
    max_value=1.0,
    value=0.3, # Дефолтное значение
    step=0.05
)

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Исходное фото")
        st.image(original_image, use_container_width=True)

    if st.button('Запустить анализ'):
       
        img_byte_arr = io.BytesIO()
        # Конвертируем в JPEG для отправки
        original_image.convert("RGB").save(img_byte_arr, format='JPEG')
        files = {'file': img_byte_arr.getvalue()}
        params = {'conf': conf_threshold}

        with st.spinner('Запрос к API...'):
            try:
                response = requests.post(API_URL, files=files, params=params)
                response.raise_for_status() 
                data = response.json()
                detections = data.get("detections", [])


                res_image = original_image.copy()
                draw = ImageDraw.Draw(res_image)
                
                table_data = []

                for det in detections:
                    box = det['bbox']
                    name = det['name']
                    conf = det['conf']
                    

                    draw.rectangle(box, outline="red", width=4)
   
                    draw.text((box[0], box[1] - 10), f"{name} {conf:.2f}", fill="red")
                    

                    table_data.append({
                        "Объект": name,
                        "Уверенность": round(conf, 3),
                        "x_min": round(box[0], 1),
                        "y_min": round(box[1], 1),
                        "x_max": round(box[2], 1),
                        "y_max": round(box[3], 1)
                    })

                with col2:
                    st.subheader("Результат")
                    st.image(res_image, use_container_width=True)

                st.divider()

                if table_data:
                    st.subheader(f"📊 Найдено объектов: {len(table_data)}")
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать результаты (.csv)", csv, "results.csv", "text/csv")
                else:
                    st.warning("Объекты не обнаружены.")

            except Exception as e:
                st.error(f"Не удалось связаться с API: {e}")