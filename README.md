# Authentix: Deepfake Detector

Authentix is a comprehensive deepfake detection platform developed during a 36-hour hackathon. It provides robust tools to identify manipulated media across 3 different domains: Images, Videos, and Audio.

## 🚀 Features

- **Image Deepfake Detection**: Uses MobileNetV2-based CNN to detect face swaps and GAN-generated images.
- **Video Deepfake Detection**: Analyzes frame sequences to identify temporal inconsistencies and manipulation.
- **Audio Deepfake Detection**: (NEW) Leverages Mel-spectrogram analysis (MFCC) and a custom CNN to distinguish real voices from AI-generated clones.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow / Keras
- **Audio Processing**: Librosa, Soundfile
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript (Cyberpunk themed)

## 📁 Repository Structure

- `app.py`: Main Flask server.
- `predict.py`: Core inference logic for all detection types.
- `preprocess.py`: Data transformation and feature extraction (Images, Video frames, Audio MFCCs).
- `train_audio_model.py`: Training script for the audio detection model.
- `model/`: Pre-trained model weights.
- `templates/`: UI components.
- `uploads/`: Temporary directory for analysis files.

## 🧪 Integration Details

For a detailed walkthrough of the recent Audio Detection integration, please refer to:
[AUDIO_INTEGRATION_WALKTHROUGH.md](AUDIO_INTEGRATION_WALKTHROUGH.md)

## 🏗️ Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application**:
   ```bash
   python app.py
   ```
3. **Access the UI**:
   Navigate to `http://127.0.0.1:5000` in your browser.

---
*Created for National Level Hackathon 2026*
