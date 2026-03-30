from fastapi.testclient import TestClient

from app.api import app
from app.dependencies import get_rag_records


def test_tools_demo_api_uses_search_chunks_tool(monkeypatch) -> None:
    async def fake_run_tools_demo_agent(question: str, records, top_k: int = 3):
        assert question == "What is Python?"
        assert top_k == 3
        return {
            "answer": "Python is a programming language.",
            "route": "tool",
            "selected_tool": "search_chunks",
            "tool_input": "What is Python?",
            "tool_output": [
                {
                    "doc_id": "doc1",
                    "title": "Python",
                    "chunk_id": "c1",
                    "text": "Python is a programming language.",
                    "score": 0.9,
                }
            ],
        }

    def fake_get_rag_records():
        return []

    monkeypatch.setattr(
        "app.routers.tools_demo.run_tools_demo_agent",
        fake_run_tools_demo_agent,
    )

    app.dependency_overrides[get_rag_records] = fake_get_rag_records

    try:
        client = TestClient(app)

        response = client.post(
            "/tools-demo/answer",
            json={
                "question": "What is Python?",
                "top_k": 3,
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "Python is a programming language."
        assert data["route"] == "tool"
        assert data["selected_tool"] == "search_chunks"
        assert data["tool_input"] == "What is Python?"
        assert isinstance(data["tool_output"], list)
        assert data["tool_output"][0]["title"] == "Python"
    finally:
        app.dependency_overrides.clear()