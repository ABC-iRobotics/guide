from guide_core.core.runtime import import_and_bind


def test_import_and_bind_module():
    ns = {}
    obj, kind, name = import_and_bind("types", namespace=ns)
    assert kind == "module"
    assert name == "types"
    assert ns[name] is obj


def test_import_and_bind_symbol_colon_syntax():
    ns = {}
    obj, kind, name = import_and_bind("types:SimpleNamespace", namespace=ns)
    assert kind in {"class", "other"}
    assert name == "SimpleNamespace"
    assert ns[name] is obj


def test_import_and_bind_symbol_dot_fallback():
    ns = {}
    obj, kind, name = import_and_bind("types.SimpleNamespace", namespace=ns)
    assert name == "SimpleNamespace"
    assert ns[name] is obj
