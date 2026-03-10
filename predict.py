import tensorflow as tf
import numpy as np


MODEL_OUTPUT_DEEPFAKE_CONFIDENCE = False
# If your model outputs a single probability score where higher means more likely 'fake':
#   True  -> [0.0 .. 1.0] in favor of fake
#   False -> [0.0 .. 1.0] in favor of real (e.g., model outputs real score)

model = None
video_model = None

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

    m = _get_model()
    if m is None:
        return "Unknown", 0.0
        
    prediction = m.predict(image, verbose=0)
    fake_score = _extract_fake_score(prediction)
    return _score_to_label_confidence(fake_score)

def predict_video_deepfake(frames):
    if len(frames) == 0:
        return "Unknown", 0.0

    v_model = _get_video_model()
    
    if v_model is not None:
        # We have an LSTM video model. It expects shape (1, num_frames, 224, 224, 3)
        # Pad or truncate frames to match whatever it expects (usually maximum 10)
        # Assuming the model was built with a fixed `max_frames` shape.
        
        # Let's inspect the expected shape from the model input:
        expected_frames = v_model.input_shape[1] # e.g. 10
        if expected_frames is not None:
            if len(frames) > expected_frames:
                frames = frames[:expected_frames]
            while len(frames) < expected_frames:
                if len(frames) > 0:
                    frames = np.concatenate([frames, [frames[-1]]])
                else:
                    frames = np.zeros((expected_frames, 224, 224, 3))
                    
        # Reshape to (1, seq_length, 224, 224, 3)
        video_sequence = np.expand_dims(frames, axis=0)
        prediction = v_model.predict(video_sequence, verbose=0)
        fake_score = _extract_fake_score(prediction)
        # The spatial model outputs 1.0 for real and 0.0 for fake.
        # But the video LSTM model outputs 1.0 for fake and 0.0 for real.
        # We must manually bypass the _score_to_label_confidence inversion.
        fake_score = float(np.clip(fake_score, 0.0, 1.0))
        if fake_score >= 0.5:
            return 'Deepfake', fake_score
        else:
            return 'Real', 1.0 - fake_score
        
    else:
        # Fallback to the old frame-by-frame approach if video model doesn't exist
        m = _get_model()
        if m is None:
            return "Unknown", 0.0
            
        predictions = m.predict(frames, verbose=0)
    
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
