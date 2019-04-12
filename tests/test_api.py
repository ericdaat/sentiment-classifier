import unittest
import json
from config import TEST_MODEL_FILEPATH
from api import create_app

app = create_app(model_filepath=TEST_MODEL_FILEPATH)
client = app.test_client()

mimetype = "application/json"
headers = {
    "Content-Type": mimetype,
    "Accept": mimetype
}


class TestClient(unittest.TestCase):
    def test_up(self):
        self.assertEqual(200, client.get("/api").status_code)

    def test_classify(self):
        res = client.post("/api/classify",
                          data=json.dumps({"text": "hi there"}),
                          headers=headers)

        self.assertEqual(200, res.status_code)

        res_json = res.get_json()
        self.assertIsInstance(res_json["score"], str)
        self.assertIn(res_json["sentiment"], {"pos", "neg"})
        self.assertIsInstance(res_json["text"], str)
