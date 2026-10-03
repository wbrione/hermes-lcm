"""Regression tests: LCM_SUMMARY_REASONING_EFFORT env knob (summary calls)."""
import pytest

from hermes_lcm.escalation import _summary_reasoning_config


def test_unset_env_sends_nothing():
    """Upstream behavior: empty env = no reasoning field on the wire."""
    import os
    old = os.environ.pop("LCM_SUMMARY_REASONING_EFFORT", None)
    try:
        assert _summary_reasoning_config() is None
    finally:
        if old is not None:
            os.environ["LCM_SUMMARY_REASONING_EFFORT"] = old


def test_low_effort_builds_enabled_config(monkeypatch):
    monkeypatch.setenv("LCM_SUMMARY_REASONING_EFFORT", "low")
    assert _summary_reasoning_config() == {"enabled": True, "effort": "low"}


def test_none_requests_reasoning_off(monkeypatch):
    monkeypatch.setenv("LCM_SUMMARY_REASONING_EFFORT", "none")
    assert _summary_reasoning_config() == {"enabled": False}


def test_case_and_whitespace_normalized(monkeypatch):
    monkeypatch.setenv("LCM_SUMMARY_REASONING_EFFORT", "  Minimal ")
    assert _summary_reasoning_config() == {"enabled": True, "effort": "minimal"}


def test_call_llm_for_summary_injects_reasoning_config(monkeypatch):
    """The call site must pass reasoning_config to the Hermes aux client."""
    import hermes_lcm.escalation as esc

    captured = {}

    class FakeResponse:
        def __init__(self):
            class _Msg:
                content = "ok summary"
            class _Choice:
                message = _Msg()
            self.choices = [_Choice()]
        usage = None

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    import sys
    import types

    # ensure the lazy import inside the function resolves to our fake
    agent_mod = sys.modules.get("agent.auxiliary_client") or types.ModuleType("agent.auxiliary_client")
    agent_mod.call_llm = fake_call_llm
    sys.modules["agent.auxiliary_client"] = agent_mod

    monkeypatch.setenv("LCM_SUMMARY_REASONING_EFFORT", "low")
    out = esc._call_llm_for_summary("summarize this", 512)
    assert captured.get("reasoning_config") == {"enabled": True, "effort": "low"}
    assert captured.get("task") == "compression"
    assert out == "ok summary"


def test_call_llm_for_summary_omits_field_when_unset(monkeypatch):
    import hermes_lcm.escalation as esc

    captured = {}

    class FakeResponse:
        def __init__(self):
            class _Msg:
                content = "ok summary"
            class _Choice:
                message = _Msg()
            self.choices = [_Choice()]

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    import sys
    import types

    agent_mod = sys.modules.get("agent.auxiliary_client") or types.ModuleType("agent.auxiliary_client")
    agent_mod.call_llm = fake_call_llm
    sys.modules["agent.auxiliary_client"] = agent_mod

    monkeypatch.delenv("LCM_SUMMARY_REASONING_EFFORT", raising=False)
    esc._call_llm_for_summary("summarize this", 512)
    assert "reasoning_config" not in captured
