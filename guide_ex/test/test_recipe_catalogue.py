"""The committed catalogue (CATALOGUE.md, catalogue.json at the package root) is a fresh one."""

from guide_ex.recipe.catalogue import render
from guide_ex.recipe.registry import PACKAGE


def test_the_catalogue_is_fresh():
    md, js = render()

    assert (PACKAGE / "CATALOGUE.md").read_text() == md, "run: python -m guide_ex.recipe catalogue"
    assert (PACKAGE / "catalogue.json").read_text() == js, "run: python -m guide_ex.recipe catalogue"
