from fastapi.testclient import TestClient

from app.api import app
from app.dependencies import get_rag_records


def test_tools_loop_demo_api(monkeypatch) -> None:
    async def fake_run_tools_loop_demo_agent(
        question: str,
        records,
        top_k: int = 3,
        max_steps: int = 3,
    ):
        assert question == "What is Python?"
        assert top_k == 3
        assert max_steps == 3

        return {
            "answer": "Python is a programming language.",
            "selected_tool": "search_chunks",
            "tool_output": "Python [c1]: Python is a programming language.",
            "steps_taken": 1,
            "next_action": "finish",
            "history": [
                {
                    "tool": "search_chunks",
                    "input": "What is Python?",
                    "output": "Python [c1]: Python is a programming language.",
                }
            ],
        }

    def fake_get_rag_records():
        return []

    monkeypatch.setattr(
        "app.routers.tools_loop_demo.run_tools_loop_demo_agent",
        fake_run_tools_loop_demo_agent,
    )

    app.dependency_overrides[get_rag_records] = fake_get_rag_records

    try:
        client = TestClient(app)

        response = client.post(
            "/tools-loop-demo/answer",
            json={
                "question": "What is Python?",
                "top_k": 3,
                "max_steps": 3,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == "Python is a programming language."
        assert data["selected_tool"] == "search_chunks"
        assert data["tool_output"] == "Python [c1]: Python is a programming language."
        assert data["steps_taken"] == 1
        assert data["next_action"] == "finish"
        assert len(data["history"]) == 1
        assert data["history"][0]["tool"] == "search_chunks"
    finally:
        app.dependency_overrides.clear()