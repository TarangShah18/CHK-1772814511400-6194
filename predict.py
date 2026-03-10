import tensorflow as tf
import numpy as np


MODEL_OUTPUT_DEEPFAKE_CONFIDENCE = False
# If your model outputs a single probability score where higher means more likely 'fake':
#   True  -> [0.0 .. 1.0] in favor of fake
#   False -> [0.0 .. 1.0] in favor of real (e.g., model outputs real score)

model = tf.keras.models.load_model('model/deepfake_model.h5')


def _extract_fake_score(predictions):
    # predictions can be shape [1], [1,1], [1,2], or [batch, 1/2]
    arr = np.array(predictions)

    if arr.ndim == 0:
        return float(arr)

    if arr.ndim == 1:
        if arr.shape[0] == 1:
            return float(arr[0])
        if arr.shape[0] == 2:
            return float(arr[1])

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return float(arr[0, 0])
        if arr.shape[1] == 2:
            return float(arr[0, 1])

    raise ValueError(f"Unsupported output shape for model predictions: {arr.shape}")


def _score_to_label_confidence(fake_score):
    # convert to deepfake probability if model outputs real score
    if not MODEL_OUTPUT_DEEPFAKE_CONFIDENCE:
        fake_score = 1.0 - fake_score

    fake_score = float(np.clip(fake_score, 0.0, 1.0))

    if fake_score >= 0.5:
        label = 'Deepfake'
        confidence = fake_score
    else:
        label = 'Real'
        confidence = 1.0 - fake_score

    return label, float(confidence)


def predict_deepfake(image):
    if image.shape != (1, 224, 224, 3):
        image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
    fake_score = _extract_fake_score(prediction)
    return _score_to_label_confidence(fake_score)

def predict_video_deepfake(frames):
    if len(frames) == 0:
        return "Unknown", 0.0

    predictions = model.predict(frames, verbose=0)

    arr = np.array(predictions)
    if arr.ndim == 1 and arr.shape[0] == len(frames):
        # Regression-style per-frame score
        fake_scores = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        fake_scores = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        # Assume class index 1 is fake
        fake_scores = arr[:, 1]
    else:
        raise ValueError(f"Unsupported output shape for video predictions: {arr.shape}")

    avg_fake_score = float(np.mean(fake_scores))
    return _score_to_label_confidence(avg_fake_score)
