from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'monkeypox_classifier.tflite'

def predict_image(file_path):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_img(file_path,target_size=(224,224))
    img = img_to_array(img,dtype=np.float32)
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    return output


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        class_labels = {
            0 : "Monkey Pox",
            1 : "Others"
        }

        output_data = predict_image(file_path)

        predicted = class_labels[np.argmax(output_data[0])]
        possibility = f"{(np.max(output_data[0])*100):.2f}%"

        data = {
            "predicted": predicted,
            "possibility": possibility,
        }

        return jsonify(data)

    return None


if __name__ == '__main__':
    app.run(debug=True)