from flask import Blueprint, jsonify

bp = Blueprint("index", __name__)


@bp.route("/", methods=("GET",))
def index():
    return jsonify(hello="there")
