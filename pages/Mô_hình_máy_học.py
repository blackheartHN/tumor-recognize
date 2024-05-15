"""
Trang streamlit dự đoán khối u não dựa trên ảnh.

Sử dụng model đã được train sẵn 'tumor-reg.keras' để phân loại ảnh
là khối u não hay không khối u não. Người dùng có thể tải lên một ảnh, và ứng dụng
sẽ hiển thị kết quả dự đoán.

Cách sử dụng:
1. Tải lên ảnh muốn dự đoán bằng cách click vào "Upload 1 ảnh".
2. Ứng dụng sẽ hiển thị ảnh đã tải lên và kết quả dự đoán.
"""
import os
import streamlit as st
import tensorflow as tf
from PIL import Image  # Import the Image module from PIL


current_dir = os.path.dirname(__file__)


model_path = os.path.join(current_dir, '../model/tumor-reg.keras')


def load_and_preprocess_image(image_file):
    """
    Tải và tiền xử lý ảnh để chuẩn bị cho việc dự đoán.
    
    Args:
        image_file: Đối tượng file ảnh đã được tải lên.
        
    Returns:
        img_array: Mảng đầu vào cho model.
    """
    img = tf.keras.preprocessing.image.load_img(image_file, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array

# Load model
model = tf.keras.models.load_model(model_path)


st.title('Dự đoán khối u não')

uploaded_file = st.file_uploader("Upload 1 ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    resized_image = image.resize((400, 400))  
    st.image(resized_image, caption='Ảnh đã được upload.', use_column_width=True)


    with st.spinner('Predicting...'):
        img_array = load_and_preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
    if prediction[0][0] > prediction[0][1]:
        st.write('Prediction: non-tumorous')
    else:
        st.write('Prediction: tumorous')
