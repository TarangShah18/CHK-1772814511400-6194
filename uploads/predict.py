import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model/deepfake_model.h5')

def predict_deepfake(image):
    if image.shape != (1, 224, 224, 3):
        image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    if prediction <= 0.5:
        label = 'Deepfake'
        confidence = (0.5 - prediction)/0.5
    else:
        label = 'Real'
        confidence = (prediction - 0.5)/0.5

    return label, float(confidence)