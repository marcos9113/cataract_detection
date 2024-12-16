from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the TFLite model
model = load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file:
        # Save the uploaded image
        image_path = 'uploads/' + file.filename
        file.save(image_path)

        # Load and preprocess the image for prediction
        img = Image.open(image_path)
        img = np.array(img.resize((94, 55)))
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        pred = model.predict(img)

        # Display the predicted class
        prediction = 'normal' if pred[0] > 0.5 else 'cataract'

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
