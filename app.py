from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
from keras.applications.inception_v3 import preprocess_input,decode_predictions
import streamlit as st

clf = load_model('inception')
img = st.file_uploader('Take any picture')
if img:
    image = Image.open(img)
    st.image(image)
    image = np.array(image)
    image = preprocess_input(image)
    image = cv2.resize(image,(299,299))
    predictions = decode_predictions(clf.predict(image.reshape(1,299,299,3)))
    st.write(predictions[0][0][1])
    
