import os
import unittest

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from main import app

load_dotenv()


class AgentIntegrationTests(unittest.TestCase):
    def test_root_reports_checkpointer(self):
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("message", payload)
        self.assertIn("checkpointer", payload)

    def test_agent_health_endpoint_reports_ok(self):
        with TestClient(app) as client:
            response = client.get("/health/agent")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["initialized"])
        self.assertEqual(payload["config_errors"], [])

    def test_chat_endpoint_returns_session_id(self):
        with TestClient(app) as client:
            response = client.post(
                "/chat/",
                json={"message": "Reply with exactly the word pong."},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload["reply"], str)
        self.assertTrue(payload["reply"].strip())
        self.assertIsInstance(payload["session_id"], str)
        self.assertTrue(payload["session_id"].strip())

    def test_chat_session_continuity(self):
        with TestClient(app) as client:
            first_response = client.post(
                "/chat/",
                json={"message": "Remember the word DRAGONFRUIT and answer only YES."},
            )
            self.assertEqual(first_response.status_code, 200)
            first_payload = first_response.json()
            session_id = first_payload["session_id"]

            second_response = client.post(
                "/chat/",
                json={
                    "message": "What word did I ask you to remember? Answer with one word only.",
                    "session_id": session_id,
                },
            )

        self.assertEqual(second_response.status_code, 200)
        second_payload = second_response.json()
        self.assertEqual(second_payload["session_id"], session_id)
        self.assertIn("dragonfruit", second_payload["reply"].strip().lower())

    def test_chat_with_document_id_routes_through_rag(self):
        document_id = os.environ["TEST_DOCUMENT_ID"]

        with TestClient(app) as client:
            response = client.post(
                "/chat/",
                json={
                    "message": "Based on this document, what is the main topic? If you cannot determine it, say so clearly.",
                    "document_id": document_id,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload["reply"], str)
        self.assertTrue(payload["reply"].strip())
        self.assertIsInstance(payload["session_id"], str)
        self.assertTrue(payload["session_id"].strip())
