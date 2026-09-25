"""TC-06 / TC-07 — the FunctionNode final security gates are non-overridable.

`FunctionNode._security_gate_input` and `._security_gate_output` are `@final`
(the framework enforces the PII scan on input and the credential scan on output).
A domain subclass that tries to override either must fail at class-definition
time with `TypeError`, and domain-specific checks must go through the sanctioned
`_extra_security_gate_input()` / `_extra_security_gate_output()` extension hooks.

Repo-agnostic: depends only on the framework surface, so it protects the gate
contract for every template regardless of its concrete node classes.
"""

import pytest

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel

# Every probe class below declares `required_trust_level` explicitly: the current wheel
# makes that mandatory in a FunctionNode subclass body, so without it the class cannot be
# defined at all and these tests fail (or pass) for a reason unrelated to the @final gates
# they are asserting on. The `match=` argument keeps TC-06/TC-07 load-bearing by requiring
# the TypeError to name the overridden gate.
from framework.schemas.agent_state import AgentState


def test_override_security_gate_input_raises_typeerror():
    """TC-06 — overriding the @final input gate raises TypeError at class def."""
    with pytest.raises(TypeError, match="_security_gate_input"):

        class BadNodeInput(FunctionNode):
            required_trust_level = TrustLevel.VERIFIED_EXTERNAL

            def execute(self, state: AgentState) -> dict:
                return {}

            def _security_gate_input(self, state: AgentState) -> AgentState:  # noqa
                return state


def test_override_security_gate_output_raises_typeerror():
    """TC-07 — overriding the @final output gate raises TypeError at class def."""
    with pytest.raises(TypeError, match="_security_gate_output"):

        class BadNodeOutput(FunctionNode):
            required_trust_level = TrustLevel.VERIFIED_EXTERNAL

            def execute(self, state: AgentState) -> dict:
                return {}

            def _security_gate_output(self, result: dict) -> dict:  # noqa
                return result


def test_extra_hooks_are_the_supported_extension_point():
    """The sanctioned domain-check hooks exist and are NOT @final — a subclass
    may override them (this is where domain PII/credential checks belong)."""

    class OkNode(FunctionNode):
        required_trust_level = TrustLevel.VERIFIED_EXTERNAL

        def execute(self, state: AgentState) -> dict:
            return {}

        def _extra_security_gate_input(self, state: AgentState) -> AgentState:
            return state

        def _extra_security_gate_output(self, result: dict) -> dict:
            return result

    # class definition succeeded (no TypeError) — the extension hooks are open.
    assert hasattr(OkNode, "_extra_security_gate_input")
    assert hasattr(OkNode, "_extra_security_gate_output")
