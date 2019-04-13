"""
We use Flask to write the API and we are using the factory pattern \
    to create the Flask application.
This is an elegant method that allows us to separate the code \
    for the app creation, and register all the blueprints in one place.

The factory runs the following steps:

- Create the Flask object
- Load the ML models and attach them
- Register the index blueprint
"""
from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
import tensorflow as tf

from sentiment_classifier.nlp.models.deep_networks import CNN


def create_app(model_filepath):
    """ Flask app factory method

    Returns:
        The created Flask application

    """
    app = Flask(__name__)

    # proxy fix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    model = CNN()  # TODO: config variable to chose the model to use
    model.load(filepath=model_filepath)

    graph = tf.get_default_graph()
    app.nlp_model = model
    app.graph = graph

    from api import index
    app.register_blueprint(index.bp)

    return app
