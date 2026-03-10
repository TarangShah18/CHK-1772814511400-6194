from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from predict import predict_deepfake, predict_video_deepfake
from preprocess import preprocess_image, preprocess_video

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_set):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_set

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
    
    if not (is_image or is_video):
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
        else:
            image = preprocess_image(filepath)
            label, confidence = predict_deepfake(image)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template('result.html', label=label, confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)