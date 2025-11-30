import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Image Classification Model')

# use a raw string or Path to avoid backslash escape problems on Windows
MODEL_PATH = Path(r"C:\Users\RONIT\OneDrive\Desktop\Animal Classification\Image_classifier.keras")
if not MODEL_PATH.exists():
	st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
else:
	model = load_model(str(MODEL_PATH))

data_cat = ['Bear',
 'Bird',
 'Cat',
 'Cow',
 'Deer',
 'Dog',
 'Dolphin',
 'Elephant',
 'Giraffe',
 'Horse',
 'Kangaroo',
 'Lion',
 'Panda',
 'Tiger',
 'Zebra']
img_height = 180
img_width = 180
image = st.text_input('Enter Image name or path', 'Bear.jpg')

# load the image, convert to array and add batch dimension correctly
try:
	image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
	img_arr = tf.keras.utils.img_to_array(image_load)   # convert PIL image to array, not array_to_img
	img_bat = np.expand_dims(img_arr, 0)

	predict = model.predict(img_bat)
	# predict may return a batch; take the first item and apply softmax
	score = tf.nn.softmax(predict[0])

	st.image(image, width=200)
	st.write('The Animal in image is ' + data_cat[int(np.argmax(score))])
	st.write('With accuracy of ' + f"{float(np.max(score) * 100):.2f}%")
except Exception as e:
	st.error(f"Failed to load or predict image: {e}")