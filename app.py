from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import traceback
from predict import predict_deepfake, predict_video_deepfake, predict_audio_deepfake
from preprocess import preprocess_image, preprocess_video, preprocess_audio

import mimetypes

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename, allowed_set):
    # Basic extension check
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_set:
        return False
        
    # Optional: We could do a basic mimetype check here but relying on extension is usually fine for initial check
    # mimetypes.guess_type(filename)
    return True

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max 100MB upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))
        
    is_image = allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS)
    is_video = allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    is_audio = allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS)
    
    if not (is_image or is_video or is_audio):
        return redirect(url_for('home'))
    
    secure_name = secure_filename(file.filename)
    if secure_name == '':
        flash('Invalid filename. Please upload a valid file.')
        return redirect(url_for('home'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    file.save(filepath)

    try:
        if is_video:
            frames = preprocess_video(filepath)
            label, confidence = predict_video_deepfake(frames)
        elif is_audio:
            features = preprocess_audio(filepath)
            label, confidence = predict_audio_deepfake(features)
        else:
            image = preprocess_image(filepath)
            label, confidence = predict_deepfake(image)
    except ValueError as e:
        flash(f'File processing error: {str(e)}')
        return redirect(url_for('home'))
    except Exception as e:
        trace = traceback.format_exc()
        flash(f'Unexpected error during analysis: {str(e)}')
        print("Prediction error:", trace)
        return redirect(url_for('home'))
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing temporary file: {e}")

    return render_template('result.html', label=label, confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    # Initialize model on startup asynchronously or let it lazy load
    # (Leaving lazy loading as default)
    app.run(debug=True)