from PIL import Image
import numpy as np
import pickle
import cv2
from keras.applications.mobilenet import preprocess_input,decode_predictions
from keras.applications import MobileNet
import streamlit as st

#pickle_in = open('mobilenet.pkl','rb')
clf = MobileNet(weights='imagenet')
img = st.file_uploader('Take any picture')
if img:
    image = Image.open(img)
    st.image(image)
    image = np.array(image)
    image = preprocess_input(image)
    image = cv2.resize(image,(224,224))
    predictions = decode_predictions(clf.predict(image.reshape(1,224,224,3)))
    st.write(predictions[0][0][1])
    
