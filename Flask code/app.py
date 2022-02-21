import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
import os

app = Flask(__name__)
model = load_model("hand-sign-images.h5")
folder = r'D:\VIT\VIT_Internships\AI-SmartBridge_IBM\Assignments\10 Flask\CNN_WebApp\uploads'
app.config['UPLOAD_FOLDER'] = folder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        x = request.files['upload']
        path = os.path.join(app.config['UPLOAD_FOLDER'], x.filename)
        x.save(path)

        img = image.img_to_array(image.load_img(path, target_size=(64,64)))
        img = np.expand_dims(img, axis=0)
        pred = np.argmax(model.predict(img))
        index = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
        pre = index[pred]
        output = 'The predicted gesture is: ' + str(pre)
    return render_template('index.html', input = x, result = output)

if __name__ == '__main__':
    app.run(debug=True)