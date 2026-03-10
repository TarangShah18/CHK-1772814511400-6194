import tensorflow as tf
import numpy as np

# Set this according to your model's output convention:
# True = model predicts probability of deepfake, False = probability of real
MODEL_OUTPUT_DEEPFAKE_CONFIDENCE = False

model = tf.keras.models.load_model('model/deepfake_model.h5')

def predict_deepfake(image):
    if image.shape != (1, 224, 224, 3):
        image = np.expand_dims(image, axis=0)

    prediction = float(model.predict(image, verbose=0)[0][0])

    if not MODEL_OUTPUT_DEEPFAKE_CONFIDENCE:
        prediction = 1.0 - prediction

    if prediction >= 0.5:
        label = 'Deepfake'
        confidence = (prediction - 0.5) / 0.5
    else:
        label = 'Real'
        confidence = (0.5 - prediction) / 0.5

    return label, float(confidence)

def predict_video_deepfake(frames):
    if len(frames) == 0:
        return "Unknown", 0.0

    # The model expects batched input e.g. (batch_size, 224, 224, 3)
    predictions = model.predict(frames, verbose=0)

    # Average the predictions across all frames
    avg_prediction = float(np.mean(predictions))

    if not MODEL_OUTPUT_DEEPFAKE_CONFIDENCE:
        avg_prediction = 1.0 - avg_prediction

    if avg_prediction >= 0.5:
        label = 'Deepfake'
        confidence = (avg_prediction - 0.5) / 0.5
    else:
        label = 'Real'
        confidence = (0.5 - avg_prediction) / 0.5

    return label, float(confidence)