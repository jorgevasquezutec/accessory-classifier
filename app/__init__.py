from flask import Flask, Response
from config import Config
from app.classifier import bp as classifier_bp
from flask_cors import CORS
from flask import send_from_directory
import os


def create_app(config_class=Config):
    app = Flask(__name__)
    #static_url_path='/images', static_folder='../data/train'
    CORS(app)
    app.config.from_object(config_class)

    # Initialize Flask extensions here

    # Register blueprints here
   
    app.register_blueprint(classifier_bp, url_prefix='/classifier')

    @app.route('/image/<path:filename>', methods=['GET'])
    def image(filename):
        directory =  os.path.join(os.path.abspath('data/train'), filename)
        # /home/jorge/Documentos/local/proyectos/products-similarity/app/../../data/train
        return send_from_directory(directory, filename+'.jpg')
            # return str(e)


    @app.route('/')
    def test_page():
        return '<h1>Testing the Flask Application Factory Pattern</h1>'

    return app
