from providers.factory import get_llm_provider, get_vlm_provider


def test_stub_vlm_provider_solves():
    vlm = get_vlm_provider()
    assert vlm.solve("q.png", {"A": "a.png"}) in {"A", "B", "C", "D"}


def test_stub_llm_provider_generates_text():
    llm = get_llm_provider()
    result = llm.generate_text({}, "A", ["some_rule"])
    assert "stem" in result
    assert "option_labels" in result
    assert "explanation" in result
