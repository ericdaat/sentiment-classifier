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
        score = current_app.nlp_model.predict([text])

        if score > 0.5:
            sentiment = "pos"
        else:
            sentiment = "neg"

    return jsonify(text=text,
                   sentiment=sentiment,
                   score="{0:0.3f}".format(score))
