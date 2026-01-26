# pip install streamlit-drawable-canvas
import streamlit as st 
from streamlit_drawable_canvas import st_canvas
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray, rgba2rgb

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

# 模型載入
model = tf.keras.models.load_model('emnist_cnn_model.keras', compile=False)

st.title('手寫英數字辨識')

# 排版成兩欄
col1, col2 = st.columns(2)

# 排版第1欄
with col1:
	# 建立畫布
	canvas_result = st_canvas(
		fill_color="rgba(0, 0, 0, 1)",	 # 畫布顏色為黑色不透明(最後一個1為不透明)
		stroke_width=10,				 # 筆畫寬度
		stroke_color="rgba(0, 0, 0, 1)", # 筆畫顏色為黑色不透明(最後一個1為不透明)
		update_streamlit=True,
		width=280,						 # 畫布寬度
		height=280,						 # 畫布高度
		drawing_mode="freedraw",
		key="canvas1",
	)

# 排版第2欄
with col2:
	# 建立按鈕
	if st.button('辨識'):
		print(canvas_result.image_data.shape)
		# 取得上傳圖片後 -> 先轉成rgb(去掉透明度) -> 再轉成灰階
		image1 = rgb2gray(rgba2rgb(canvas_result.image_data))	
		image_resized = resize(image1, (28, 28))  
		X1 = image_resized.reshape(1,28,28) # / 255
		X1 = np.abs(1-X1)
		
		st.write("predict...")
		predictions = np.argmax(model.predict(X1), axis=-1)
		
		# EMNIST 的 By_Class 資料集有62個種類，編號:0~9(數字)、10~35(大寫a~z)、36~61(小寫A~Z)
		token = predictions[0]
		match token:
		  case token if token >= 0 and token < 10:
			  st.write('# ' + str(token))
		  case token if token >= 10 and token < 36:
			  st.write('# ' + chr(ord('A') + token - 10))
		  case token if token >= 36 and token < 62:
			  st.write('# ' + chr(ord('a') + token - 36))
		  case _:
			  st.write('# 預測錯誤!')
			
		st.image(image_resized)
