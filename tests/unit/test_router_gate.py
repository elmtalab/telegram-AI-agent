from apps.router import gate

def test_gate_allows():
    assert gate.gate_request({})
