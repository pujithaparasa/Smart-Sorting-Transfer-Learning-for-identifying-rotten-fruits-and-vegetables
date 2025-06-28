# app.py

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

# Load label mappings
with open("labels.json", "r") as f:
    class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    class_name = labels[class_index]
    confidence = prediction[class_index] * 100
    return f"{class_name} ({confidence:.2f}%)"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        image = request.files["image"]
        if image:
            filepath = os.path.join("static/uploads", image.filename)
            image.save(filepath)
            result = predict_image(filepath)
            return render_template("predict.html", prediction=result, image_path=filepath)
    return render_template("predict.html", prediction=result)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
