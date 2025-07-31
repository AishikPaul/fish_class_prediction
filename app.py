from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = Flask(__name__)

# Load the trained model once
# model = load_model('best_model.h5')
# Set up the TFLite interpreter once (globally)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Replace with your real class names
class_names = sorted([
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch',
    'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish',
    'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
    'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
    'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
])

# Predict from in-memory image
def predict_image(file):
    # Load and preprocess the image
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # Encode image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return predicted_class, confidence, img_b64



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_data = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            prediction, confidence, img_data = predict_image(file)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_data=img_data)


if __name__ == "__main__":
    app.run(debug=True)
