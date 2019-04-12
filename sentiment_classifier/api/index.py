""" Index blueprint for the Flask API.
This blueprint hosts the code for classifying a sequence.
"""

from flask import Blueprint, jsonify, request, current_app
from nlp.preprocessing import clean_text

bp = Blueprint("index", __name__)


@bp.route("/api", methods=("GET",))
def index():
    """ api status check method.

    Returns:
        json
    """
    return jsonify(status="running")


@bp.route("/api/classify", methods=("POST",))
def classify():
    """ api method for predicting the sentiment on a given sequence.

    Returns:
        A json with text, sentiment and score

    """
    data = request.get_json()
    text = data.get("text")

    with current_app.graph.as_default():
        predictions = current_app.nlp_model.predict(
            texts=[text],
            preprocessing_function=clean_text
        )
        score = predictions[0][0]

        if score > 0.5:
            sentiment = "pos"
        else:
            sentiment = "neg"

    return jsonify(
        text=text,
        sentiment=sentiment,
        score="{0:0.3f}".format(score)
    )
