import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image
import os

#Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Loading the model
model = load_model('CNN_model.h5')

#Load the class names
class_indices = np.load('class_names.npy', allow_pickle=True).item()
class_names = {v: k for k, v in class_indices.items()}

test_image = r'C:\\Users\\SAGAR DEEP\\Desktop\\herbal_project_all\\herbalcode\\leafimg\\Balloon Vine\\003_9.jpg'

#Loading and Preprocessing the image
target_size = (150, 150)
img = image.load_img(test_image, target_size=target_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

#Making a prediction
predictions = model.predict(img_array)

#Identify the predicted class
pred_class_index = np.argmax(predictions)
pred_class_name = class_names[pred_class_index]

# Print the predicted class and confidence score
confidence = predictions[0][pred_class_index] * 100 #calculates as percentage
print(f"Predicted class: {pred_class_name} with confidence: {confidence:.2f}%")

