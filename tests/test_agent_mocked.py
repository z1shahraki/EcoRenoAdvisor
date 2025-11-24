import agent.agent as agent_module


def test_agent_returns_mock_response(monkeypatch):
    """Ensure the agent can run end-to-end with mocked dependencies."""
    fake_materials = [
        {
            "name": "Mock Bamboo Flooring",
            "category": "flooring",
            "price_per_m2": 70,
            "eco_score": 0.9,
            "voc_level_num": 1,
        }
    ]
    fake_docs = ["Mock document chunk about flooring"]
    fake_response = "This is a dummy recommendation for testing."

    monkeypatch.setattr(agent_module, "filter_materials", lambda **_: fake_materials)
    monkeypatch.setattr(agent_module, "rag_search", lambda *args, **kwargs: fake_docs)

    def mock_call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        return fake_response

    monkeypatch.setattr(
        agent_module.RenovationAgent, "call_llm", mock_call_llm, raising=False
    )

    result = agent_module.agent(
        query="Suggest flooring",
        filters={"category": "flooring", "max_price": 100},
    )

    assert isinstance(result, str)
    assert fake_response in result

