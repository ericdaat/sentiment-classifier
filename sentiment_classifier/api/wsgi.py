from api import create_app
from waitress import serve
from config import PROD_MODEL_FILEPATH

if __name__ == "__main__":
    application = create_app(model_filepath=PROD_MODEL_FILEPATH)
    serve(application, port=8000)
