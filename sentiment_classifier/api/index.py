from flask import Blueprint, jsonify, request, current_app

bp = Blueprint("index", __name__)


@bp.route("/api", methods=("GET",))
def index():
    return jsonify(hello="there")


@bp.route("/api/classify", methods=("POST",))
def classify():
    data = request.get_json()
    text = data.get("text")

    with current_app.graph.as_default():
        prediction = current_app.nlp_model.predict([text])
        proba = prediction[0][0]

        if proba > 0.5:
            sentiment = "pos"
        else:
            sentiment = "neg"

    return jsonify(text=text,
                   sentiment=sentiment,
                   score=str(proba))
