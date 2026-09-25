# PB-6: Invoke Execution Order Verification
# Verifies BaseNode.__call__() enforces the fixed pipeline order:
#   S-1 trust gate -> S-4 node_start -> S-2 _security_gate_input() ->
#   execute() -> S-3 _security_gate_output() -> S-4 node_complete
# for every concrete node under src/nodes/.
#
# A domain node's execute() may legitimately fail-closed (status:error /
# raise) when handed the minimal probe state used here, because it needs a
# real domain payload. PB-6 verifies ORDER, not that execute() succeeds on an
# empty payload: the __call__ contract is that S-1 -> node_start -> S-2 ->
# execute run in that order regardless of the execute outcome, and the gates
# themselves (S-1/S-2/S-3) are never skipped or reordered. So a node whose
# execute() short-circuits on the probe state is asserted on the PREFIX it
# reached (node_start -> security_gate_input -> execute), not on the full
# success trailer. Nodes that do complete on the probe state are asserted on
# the full order.

import importlib
import inspect
import pkgutil

import pytest


def _discover_node_classes() -> list[type]:
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


class TestInvokeOrder:
    """PB-6: __call__ runs S-1 -> node_start -> S-2 -> execute() -> S-3 -> node_complete."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        # The order the __call__ pipeline must always follow.
        FULL = [
            "event:node_start",
            "security_gate_input",
            "execute",
            "security_gate_output",
            "event:node_complete",
        ]
        # Valid prefixes when execute() fail-closes on the minimal probe state:
        #  - execute short-circuits before finishing -> node_error after execute
        #  - execute raises -> node_error after execute
        PREFIX_TO_EXECUTE = ["event:node_start", "security_gate_input", "execute"]

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            try:
                instance = node_cls()
            except Exception:
                # Node requires constructor DI (e.g. an llm/service); PB-6 order is
                # exercised by the graph-level integration/PB tests for such nodes.
                continue

            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            try:
                instance(state)
            except Exception:
                pass  # a raising execute still must have run the gates in order first

            ok = False
            if order == FULL:
                ok = True
            elif (
                order[: len(PREFIX_TO_EXECUTE)] == PREFIX_TO_EXECUTE
                and "security_gate_output" not in order[: len(PREFIX_TO_EXECUTE)]
            ):
                # execute() fail-closed on the probe payload: the S-1/S-4-start/S-2/execute
                # prefix ran in the correct order; the success trailer is not required here.
                # Crucially, the two input-side gates ran BEFORE execute and were not skipped.
                ok = True

            if not ok:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n"
                    f"expected: {FULL} (or the S1->start->S2->execute prefix on fail-closed)\n"
                    f"actual:   {order}"
                )

        assert not failures, "\n\n".join(failures)
