import tensorflow as tf
import numpy as np
import cv2

MODEL_OUTPUT_DEEPFAKE_CONFIDENCE = False

model = None
video_model = None
audio_model = None

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def _get_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model('model/deepfake_model.h5')
        except:
            print("Warning: Could not load image model")
            return None
    return model


def _get_video_model():
    global video_model
    if video_model is None:
        try:
            video_model = tf.keras.models.load_model('model/deepfake_video_model.h5')
        except:
            print("Warning: Could not load video model, falling back to image model")
            return None
    return video_model

def _get_audio_model():
    global audio_model
    if audio_model is None:
        try:
            audio_model = tf.keras.models.load_model('model/deepfake_audio_model.h5')
        except:
            print("Warning: Could not load audio model")
            return None
    return audio_model


def _extract_fake_score(predictions):
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


# -------------------------
# IMAGE PREDICTION
# -------------------------

def predict_deepfake(image):
    if image.shape != (1, 224, 224, 3):
        image = np.expand_dims(image, axis=0)
        
    m = _get_model()

    if m is None:
        return "Unknown", 0.0

    # Predict directly on the preprocessed image
    # The model was trained on full 224x224 images (no face extraction)
    prediction = m.predict(image, verbose=0)

    fake_score = _extract_fake_score(prediction)

    return _score_to_label_confidence(fake_score)


# -------------------------
# VIDEO PREDICTION
# -------------------------

def predict_video_deepfake(frames):

    if len(frames) == 0:
        return "Unknown", 0.0

    v_model = _get_video_model()

    if v_model is not None:

        expected_frames = v_model.input_shape[1]

        if expected_frames is not None:
            if len(frames) > expected_frames:
                frames = frames[:expected_frames]

            while len(frames) < expected_frames:
                frames = np.concatenate([frames, [frames[-1]]])

        video_sequence = np.expand_dims(frames, axis=0)

        prediction = v_model.predict(video_sequence, verbose=0)

        fake_score = _extract_fake_score(prediction)

        fake_score = float(np.clip(fake_score, 0.0, 1.0))

        if fake_score >= 0.5:
            return 'Deepfake', fake_score
        else:
            return 'Real', 1.0 - fake_score

    else:

        m = _get_model()

        if m is None:
            return "Unknown", 0.0

        predictions = m.predict(frames, verbose=0)

        arr = np.array(predictions)

        if arr.ndim == 1 and arr.shape[0] == len(frames):
            fake_scores = arr

        elif arr.ndim == 2 and arr.shape[1] == 1:
            fake_scores = arr[:, 0]

        elif arr.ndim == 2 and arr.shape[1] == 2:
            fake_scores = arr[:, 1]

        else:
            raise ValueError(f"Unsupported output shape for video predictions: {arr.shape}")

        avg_fake_score = float(np.mean(fake_scores))

        return _score_to_label_confidence(avg_fake_score)

# -------------------------
# AUDIO PREDICTION
# -------------------------

def predict_audio_deepfake(features):
    m = _get_audio_model()
    if m is None:
        return "Unknown", 0.0

    if features.ndim == 3:
        features = np.expand_dims(features, axis=0)

    prediction = m.predict(features, verbose=0)
    fake_score = _extract_fake_score(prediction)

    return _score_to_label_confidence(fake_score)