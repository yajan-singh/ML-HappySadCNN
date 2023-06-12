import os
import numpy as np
import tensorflow as tf
import cv2
CLASSES = []
for dir in os.listdir('dataset'):
    if dir[0] == '.':
        continue
    CLASSES.append(dir)

CLASSES = sorted(CLASSES)

# Load Model
model = tf.keras.models.load_model('model.h5')

# Load Image
img = cv2.imread('test.jpg')

# Resize Image
img = tf.image.resize(img, (256, 256))

# Predict Image
prediction = model.predict(np.expand_dims(img/255, 0))[0]
print(f'Predicted emotion: {CLASSES[np.argmax(prediction)]}')
