import pytest

from app.agents.tools_loop_demo.runtime import run_tools_loop_demo_agent
from app.agents.tools_loop_demo.schemas import ToolDecision


def build_fake_tool_configs() -> dict[str, dict]:
    return {
        "calculator": {
            "node": "tool",
            "kind": "math",
            "input_mode": "tool_input",
            "callable": lambda expression: "12",
        },
        "search_chunks": {
            "node": "tool",
            "kind": "retrieval",
            "input_mode": "question+records",
            "callable": lambda question, records, top_k=3: (
                "Python [c1]: Python is a programming language."
            ),
        },
        "list_docs": {
            "node": "tool",
            "kind": "metadata",
            "input_mode": "records",
            "callable": lambda records: "Python (doc1), FastAPI (doc2)",
        },
    }


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_uses_calculator_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["calculator"]

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> ToolDecision:
        return ToolDecision(
            tool=decisions.pop(0),
            reason="test decision",
        )

    async def fake_assess_whether_to_continue_with_llm(
        question: str,
        history_text: str,
        steps_taken: int,
        max_steps: int,
    ) -> str:
        return "finish"

    async def fake_final_llm(prompt: str) -> str:
        return "The result is 12."

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.assess_whether_to_continue_with_llm",
        fake_assess_whether_to_continue_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    result = await run_tools_loop_demo_agent("2 + 2 * 5")

    assert result["selected_tool"] == "calculator"
    assert result["tool_output"] == "12"
    assert result["answer"] == "The result is 12."
    assert result["steps_taken"] == 1
    assert result["next_action"] == "finish"


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_two_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["search_chunks"]
    tool_configs = build_fake_tool_configs()

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> ToolDecision:
        return ToolDecision(
            tool=decisions.pop(0),
            reason="test decision",
        )

    async def fake_assess_whether_to_continue_with_llm(
        question: str,
        history_text: str,
        steps_taken: int,
        max_steps: int,
    ) -> str:
        return "finish"

    async def fake_final_llm(prompt: str) -> str:
        return "Python is a programming language."

    def fake_get_tool_config(tool_name: str) -> dict | None:
        return tool_configs.get(tool_name)

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.get_tool_config",
        fake_get_tool_config,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.assess_whether_to_continue_with_llm",
        fake_assess_whether_to_continue_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    records = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "score": 0.9,
            "embedding": [0.1],
        }
    ]

    result = await run_tools_loop_demo_agent(
        "What is Python?",
        records=records,
    )

    assert result["steps_taken"] == 1
    assert result["selected_tool"] == "search_chunks"
    assert result["next_action"] == "finish"
    assert "Python" in result["answer"]


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_accumulates_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["calculator"]

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> ToolDecision:
        return ToolDecision(
            tool=decisions.pop(0),
            reason="test decision",
        )

    async def fake_assess_whether_to_continue_with_llm(
        question: str,
        history_text: str,
        steps_taken: int,
        max_steps: int,
    ) -> str:
        return "finish"

    async def fake_final_llm(prompt: str) -> str:
        assert "Tool history:" in prompt
        assert "Tool: calculator" in prompt
        assert "Output:\n12" in prompt
        return "The calculation result is 12."

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.assess_whether_to_continue_with_llm",
        fake_assess_whether_to_continue_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    result = await run_tools_loop_demo_agent("2 + 2 * 5")

    assert result["steps_taken"] == 1
    assert len(result["history"]) == 1
    assert result["history"][0]["tool"] == "calculator"
    assert result["history"][0]["input"] == "2 + 2 * 5"
    assert result["history"][0]["output"] == "12"
    assert result["answer"] == "The calculation result is 12."
    assert result["next_action"] == "finish"


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_runs_multi_tool_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["list_docs", "search_chunks"]
    assess_decisions = ["continue", "finish"]
    tool_configs = build_fake_tool_configs()

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> ToolDecision:
        return ToolDecision(
            tool=decisions.pop(0),
            reason="test decision",
        )

    async def fake_assess_whether_to_continue_with_llm(
        question: str,
        history_text: str,
        steps_taken: int,
        max_steps: int,
    ) -> str:
        return assess_decisions.pop(0)

    async def fake_final_llm(prompt: str) -> str:
        assert "Tool: list_docs" in prompt
        assert "Tool: search_chunks" in prompt
        return "Python is documented in the available sources and is a programming language."

    def fake_get_tool_config(tool_name: str) -> dict | None:
        return tool_configs.get(tool_name)

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.assess_whether_to_continue_with_llm",
        fake_assess_whether_to_continue_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.get_tool_config",
        fake_get_tool_config,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    records = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "score": 0.9,
            "embedding": [0.1],
        },
        {
            "doc_id": "doc2",
            "title": "FastAPI",
            "chunk_id": "c2",
            "text": "FastAPI is a Python framework.",
            "score": 0.8,
            "embedding": [0.2],
        },
    ]

    result = await run_tools_loop_demo_agent(
        "What documents are available about Python?",
        records=records,
        max_steps=3,
    )

    assert result["steps_taken"] == 2
    assert len(result["history"]) == 2
    assert result["history"][0]["tool"] == "list_docs"
    assert result["history"][1]["tool"] == "search_chunks"
    assert result["next_action"] == "finish"
    assert "Python" in result["answer"]


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_uses_llm_assessment_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["search_chunks"]
    tool_configs = build_fake_tool_configs()

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> ToolDecision:
        return ToolDecision(
            tool=decisions.pop(0),
            reason="test decision",
        )

    async def fake_assess_whether_to_continue_with_llm(
        question: str,
        history_text: str,
        steps_taken: int,
        max_steps: int,
    ) -> str:
        assert "Tool: search_chunks" in history_text
        return "finish"

    async def fake_final_llm(prompt: str) -> str:
        return "Python is a programming language."

    def fake_get_tool_config(tool_name: str) -> dict | None:
        return tool_configs.get(tool_name)

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.get_tool_config",
        fake_get_tool_config,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.assess_whether_to_continue_with_llm",
        fake_assess_whether_to_continue_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    result = await run_tools_loop_demo_agent(
        "What is Python?",
        records=[],
        max_steps=3,
    )

    assert result["steps_taken"] == 1
    assert result["next_action"] == "finish"
    assert "Python" in result["answer"]