import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import pandas as pd
import numpy as np
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('X:/DS minor project/Models/Resnet50.h5')
    return model
model = load_model()
st.write("""
# Paddy Disease Classification
""")
file = st.file_uploader("Upload image of paddy here",type=["jpg","png"])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size = (400,400)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None :
    st.text("No image uploaded")
else :
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    class_names = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight','blast','brown_spot','dead_heart','downy_mildew',
                   'hispa','normal','tungro']
    string = "Paddy is classified as : "+class_names[np.argmax(predictions)]
    st.success(string)