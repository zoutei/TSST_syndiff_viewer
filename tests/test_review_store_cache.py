from review.app import (
    _store_payload_cache,
    _store_payload_key,
    clear_store_payload_cache,
)


def test_store_payload_cache_clear():
    clear_store_payload_cache()
    key = _store_payload_key("event_a", "ws", "lc_prf_on_diffs", "primary")
    _store_payload_cache[key] = ({"event": "event_a"}, "OK", {"color": "#2e7d32"})
    assert key in _store_payload_cache
    clear_store_payload_cache()
    assert key not in _store_payload_cache
