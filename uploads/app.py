from flask import Flask, render_template, request, redirect, url_for
import os
from predict import predict_deepfake
from preprocess import preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def defect():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = preprocess_image(filepath)
    label, confidence = predict_deepfake(image)

    return render_template('result.html', label=label, confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
    
