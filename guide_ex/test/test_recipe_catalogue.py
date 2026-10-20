"""The committed catalogue (CATALOGUE.md, catalogue.json at the package root) is a fresh one."""

from guide_ex.recipe.catalogue import render
from guide_ex.recipe.registry import PACKAGE


def test_the_catalogue_is_fresh():
    md, js = render()

    assert (PACKAGE / "CATALOGUE.md").read_text() == md, "run: python -m guide_ex.recipe catalogue"
    assert (PACKAGE / "catalogue.json").read_text() == js, "run: python -m guide_ex.recipe catalogue"


def test_a_leaf_registered_from_outside_the_package_is_skipped(monkeypatch):
    from guide_ex.core.base_node import BaseNode
    from guide_ex.recipe.registry import default_registry, load_defaults

    class Outside(BaseNode):
        def run(self):
            pass

    load_defaults()
    monkeypatch.setitem(default_registry._entries, "Outside", (Outside, "test"))

    assert "| Outside |" not in render()[0]
