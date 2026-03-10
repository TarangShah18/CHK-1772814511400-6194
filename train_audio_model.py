import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
from preprocess import preprocess_audio

def build_audio_model(input_shape=(40, 300, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_mock_audio_dataset(target_dir="dataset_audio"):
    os.makedirs(os.path.join(target_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "fake"), exist_ok=True)
    
    # In a real scenario, we'd download the Kaggle dataset.
    # For this demo, we'll create some synthetic noise files to test the pipeline.
    import soundfile as sf
    
    print("Generating Mock Audio Dataset...")
    for i in range(100):
        # Real: Lower frequency noise
        real_audio = np.random.normal(0, 0.1, 16000 * 2) # 2 seconds
        sf.write(os.path.join(target_dir, "real", f"real_{i}.wav"), real_audio, 16000)
        
        # Fake: Higher frequency noise
        fake_audio = np.random.normal(0, 0.2, 16000 * 2)
        sf.write(os.path.join(target_dir, "fake", f"fake_{i}.wav"), fake_audio, 16000)
    print("Mock audio dataset generated.")

def train_audio_model(dataset_dir="dataset_audio", model_path="model/deepfake_audio_model.h5"):
    X = []
    y = []
    
    for label, category in enumerate(["fake", "real"]):
        dir_path = os.path.join(dataset_dir, category)
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(dir_path, filename)
                try:
                    features = preprocess_audio(filepath)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    model = build_audio_model(input_shape=X.shape[1:])
    model.fit(X, y, epochs=20, batch_size=8)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Audio model saved to {model_path}")

if __name__ == "__main__":
    generate_mock_audio_dataset()
    train_audio_model()
