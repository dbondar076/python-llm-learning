import pytest

from app.agents.tools_loop_demo.runtime import run_tools_loop_demo_agent


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_uses_calculator_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["calculator", "finish"]

    async def fake_decide_next_tool_with_llm(
        question: str,
        steps_taken: int,
        max_steps: int,
        previous_tool_output: str | None,
    ) -> str:
        return decisions.pop(0)

    async def fake_final_llm(prompt: str) -> str:
        return "The result is 12."

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
    )
    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.run_text_prompt_with_retry_async",
        fake_final_llm,
    )

    result = await run_tools_loop_demo_agent("2 + 2 * 5")

    assert result["selected_tool"] == "finish"
    assert result["tool_output"] == "12"
    assert result["answer"] == "The result is 12."
    assert result["steps_taken"] == 1


@pytest.mark.asyncio
async def test_tools_loop_demo_agent_two_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = ["search_chunks", "finish"]

    async def fake_decide_next_tool_with_llm(*args, **kwargs):
        return decisions.pop(0)

    async def fake_final_llm(prompt: str) -> str:
        return "Python is a programming language."

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.nodes.decide_next_tool_with_llm",
        fake_decide_next_tool_with_llm,
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
    assert "Python" in result["answer"]