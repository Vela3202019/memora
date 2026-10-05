import pytest

from evals.run_extraction_eval import require_cross_family_judge, resolve_extractor_model


def test_subscription_backend_defaults_to_a_claude_model():
    assert resolve_extractor_model("claude-subscription", None) == "claude-sonnet-5"


def test_openai_backend_keeps_its_default_model():
    assert resolve_extractor_model("openai-compatible", None) == "gpt-5.2"


def test_explicit_model_wins_over_the_backend_default():
    assert resolve_extractor_model("claude-subscription", "claude-opus-5-5") == "claude-opus-5-5"


@pytest.mark.parametrize("backend", ["claude-subscription", "anthropic"])
def test_claude_extractor_cannot_be_judged_by_claude(backend):
    with pytest.raises(SystemExit, match="same model family"):
        require_cross_family_judge(backend, "claude-subscription")


def test_cross_family_pairs_are_accepted():
    require_cross_family_judge("claude-subscription", "deepseek")
    require_cross_family_judge("openai-compatible", "claude-subscription")
