from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
import tensorflow as tf

from nlp import ml


def create_app():
    """ Flask app factory method

    Returns:
        The created Flask application

    """
    app = Flask(__name__)

    # proxy fix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    model = ml.CNN()  # TODO: config variable to chose the model to use
    model.load()

    graph = tf.get_default_graph()
    app.nlp_model = model
    app.graph = graph

    from api import index
    app.register_blueprint(index.bp)

    return app
