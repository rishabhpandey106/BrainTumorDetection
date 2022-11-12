import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)

model=load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check ')

def get_className (classNo):
    if classNo==0:
        return "No Brain Tumor"
    elif classNo==1:
        return "Yes Brain Tumor"

def getResult(img):
    image=cv2.imread(img)
    image=Image.fromarray(image, 'RGB')
    image=image.resize((64, 64))
    image=np.array(image)
    input_img=np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods =['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename (f. filename) )
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)

# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(64,64)) #target_size must agree with what the trained model expects!!

#     # Preprocessing the image
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img.astype('float32')/255
   
#     preds = model.predict(img)

   
   
#     pred = np.argmax(preds,axis = 1)
#     return pred

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         pred = model_predict(file_path, model)
#         os.remove(file_path)#removes file from the server after prediction has been returned

#         # Arrange the correct return according to the model. 
# 		# In this model 1 is Pneumonia and 0 is Normal.
#         # str0 = 'No Tumor'
#         # str1 = 'Tumor'
#         if pred == 0:
#             return "NO BRAIN TUMOR"
#         elif pred ==1:
#             return "YES BRAIN TUMOR"
#     return None

#     #this section is used by gunicorn to serve the app on Heroku
# if __name__ == '__main__':
#         app.run(debug=True, host="localhost", port=8080)


# import keras
# import numpy as np
# import streamlit as st
# from image_classification import teachable_machine_classification
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array, load_img
# from PIL import Image, ImageOps

# st.title("Brain Tumor or Healthy Brain")
# st.header("Brain Tumor MRI Classifier ")
# st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")
# uploaded_file = st.file_uploader(
#     "Choose an image.... ", type=["jpg", "png", "jpeg"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_File)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("Classifying.. .")
#     st.write("")
#     label = teachable_machine_classification(image, 'BrainTumor10Epochs.h5')
#     if label == 0:
#         st.write("The MRI scan detects a brain tumor")
#     else:
#         st.write("The MRI scan shows an healthy brain")
# https://teachablemachine.withgoogle.com/models/ZH76p1gDe/

# from teachable_machine import TeachableMachine

# my_model = TeachableMachine(model_path='BrainTumor10Epochs.h5', model_type='h5')

# img_path = 'F:\\dlp\\pred\\pred0.jpg'
# img=img_path.resize((64,64))

# result = my_model.classify_image(img_path)
# print(result)
