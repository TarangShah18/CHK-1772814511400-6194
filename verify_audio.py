import os
import numpy as np
from preprocess import preprocess_audio
from predict import predict_audio_deepfake

def test_audio_pipeline():
    print("Starting audio pipeline verification...")
    
    # Use one of the mock files generated during training
    test_file = "dataset_audio/real/real_0.wav"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found. Run train_audio_model.py first.")
        return
        
    try:
        print(f"Testing with file: {test_file}")
        features = preprocess_audio(test_file)
        print(f"Preprocessed features shape: {features.shape}")
        
        label, confidence = predict_audio_deepfake(features)
        print(f"Prediction Result: Label={label}, Confidence={confidence:.2%}")
        
        print("\nVerification successful!")
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_pipeline()
