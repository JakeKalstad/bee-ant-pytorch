from webapp import MLFlask
from flask import request, jsonify, render_template
from webapp.helpers import get_prediction
from flask_cors import cross_origin, logging

app = MLFlask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.getLogger('flask_cors').level = logging.DEBUG


@app.route('/')
def hello():
    return render_template('base.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    print("app")
    if request.method == 'POST':
        print(request)
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(app, image_bytes=img_bytes)
        print(class_id.item())

        print(class_name)
        response = jsonify(
            {'class_id': class_id.item(), 'class_name': class_name})

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
