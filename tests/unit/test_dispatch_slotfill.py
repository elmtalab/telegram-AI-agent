from apps.dispatch_controller import slotfill

def test_slotfill_passthrough():
    req = {"name": "test"}
    assert slotfill.fill_slots(req) == req
