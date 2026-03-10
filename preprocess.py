import cv2
import numpy as np

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def preprocess_video(video_path, max_frames=10):
    # NOTE: audio file update is pending.
    # Future extension: if the input video has an audio track, extract and process audio features in parallel.
    # This is the requested comment placeholder for "audio file update".
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return np.array([])
            
        # Calculate step size to evenly sample max_frames
        step = max(1, total_frames // max_frames)
        
        count = 0
        extracted = 0
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                extracted += 1
                
            count += 1
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("Could not extract any valid frames from the video.")
            
        return np.array(frames)
        
    except Exception as e:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        raise ValueError(f"Video preprocessing failed: {str(e)}")