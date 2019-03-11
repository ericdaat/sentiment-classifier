from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
import tensorflow as tf

from nlp import ml


def create_app():
    app = Flask(__name__)

    # proxy fix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    model = ml.CNN()
    model.load()

    graph = tf.get_default_graph()
    app.nlp_model = model
    app.graph = graph

    from api import index
    app.register_blueprint(index.bp)

    return app
