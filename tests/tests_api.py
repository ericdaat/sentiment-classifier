import unittest
import json
from api import create_app

app = create_app()
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
