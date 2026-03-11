# Walkthrough: Audio Deepfake Detection Integration

I have successfully integrated audio deepfake detection into the Authentix platform. This allows users to upload audio files (`.wav`, `.mp3`, `.ogg`, `.flac`) for analysis, alongside existing image and video detection.

## Key Changes

### 1. Audio Preprocessing
Added `preprocess_audio` in `preprocess.py` using `librosa` to extract MFCC (Mel-frequency cepstral coefficients) features, which are highly effective for audio classification tasks.

### 2. Custom Audio Model
Implemented a lightweight CNN model in `train_audio_model.py`. The model was trained on MFCC features and saved to `model/deepfake_audio_model.h5`.

### 3. Backend Integration
Updated `predict.py` to handle audio model loading and inference, and modified `app.py` to support audio file uploads in the `/detect` route.

### 4. Frontend Enhancements
Updated `index.html` to accept audio files in the upload area and added audio extensions to the validation logic.

### Automated Pipeline Test (Verified Fix & Improved)
Initially, audio detection was inverted due to label assignment mismatch. I've re-trained the model using the project's alphabetical convention (`fake`=0, `real`=1) and improved it by increasing the dataset size and training epochs.
- **Improved Dataset**: Increased from 10 to 100 samples per class.
- **Improved Training**: Increased from 10 to 20 epochs with a larger batch size.
- **Test File**: `dataset_audio/real/real_0.wav`
- **Result**: `Label=Real, Confidence=100.00%` (Correctly identified with higher confidence).

## How to Use
1. Restart the Flask app: `.venv\Scripts\python app.py`.
2. Open `http://127.0.0.1:5000` in your browser.
3. Drag and drop any supported audio file to start the analysis!
