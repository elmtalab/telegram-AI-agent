from apps.orchestrator import idempotency

def test_idempotent():
    data = {"a": 1}
    assert idempotency.ensure_idempotent(data) == data
