import os
import logging


# Constants

API_PORT = 8000

PROD_MODEL_FILEPATH = os.path.join(".", "bin")
TEST_MODEL_FILEPATH = os.path.join("/", "tmp", "sentiment-classifier")

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
