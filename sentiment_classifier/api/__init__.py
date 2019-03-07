from flask import Flask
from werkzeug.contrib.fixers import ProxyFix

from nlp import ml


def create_app():
    app = Flask(__name__)

    # proxy fix
    app.wsgi_app = ProxyFix(app.wsgi_app)

    model = ml.LogisticRegression()
    model.load()

    app.nlp_model = model

    from api import index
    app.register_blueprint(index.bp)

    return app
