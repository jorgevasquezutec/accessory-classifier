from app.classifier import bp
from flask import request, Response
import json
from app.classifier import image_classifier
from flask import send_from_directory
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file_request(f):
    def wrap(*args, **kwargs):
        if 'file' not in request.files:
            message = {'message': 'No file part!'}
            json_msg = json.dumps(message)
            return Response(json_msg, status=401, mimetype="application/json")

        file = request.files['file']
        user_id = request.form.get('user_id')

        if file.filename == '':
            message = {'message': 'No image selected for uploading'}
            json_msg = json.dumps(message)
            return Response(json_msg, status=401, mimetype="application/json")

        if (user_id == ''):
            message = {'message': 'No user id selected for uploading'}
            json_msg = json.dumps(message)
            return Response(json_msg, status=401, mimetype="application/json")

        if file and not allowed_file(file.filename):
            message = {'message': 'Allowed image types are ->' +
                       str(ALLOWED_EXTENSIONS)}
            json_msg = json.dumps(message)
            return Response(json_msg, status=401, mimetype="application/json")

        return f(*args, **kwargs)

    wrap.__name__ = f.__name__
    return wrap


@bp.route('/')
def index():
    return '<h1>Testing the Clasifier Object API</h1>'


@bp.route('/image/<path:filename>', methods=['GET'])
def image(filename):
    directory = os.path.join(bp.root_path, '../../data/train')
    print(directory)
    return send_from_directory(directory, filename)


@bp.route('/predict', methods=['POST'])
@file_request
def predict():
    try:
        file = request.files['file']
        # faceRecognizer.FaceRecognizer().insert_face(user_id, file)
        predicted_classes = image_classifier.ImageClassifier(
            'db/model.pth', 'db/classes.txt').predict(file)
        #print(predicted_class)
        # current url
        retornar_message = []
        for predicted_class in predicted_classes:
            url = request.url_root + 'image/' + predicted_class
            message = {'message': 'Predicted class',
                       'class': predicted_class,
                       'url': url
                       }
            retornar_message.append(message)
        json_msg = json.dumps(retornar_message)
        return Response(json_msg, status=200, mimetype="application/json")
    except Exception as e:
        print(e)
        message = {'message': str(e)}
        json_msg = json.dumps(message)
        return Response(json_msg, status=401, mimetype="application/json")
