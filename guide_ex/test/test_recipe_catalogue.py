"""The committed catalogue (recipes/CATALOGUE.md, recipes/catalogue.json) is a fresh one."""

from guide_ex.recipe.catalogue import render
from guide_ex.recipe.registry import LIBRARY


def test_the_catalogue_is_fresh():
    md, js = render()

    assert (LIBRARY / "CATALOGUE.md").read_text() == md, "run: python -m guide_ex.recipe catalogue"
    assert (LIBRARY / "catalogue.json").read_text() == js, "run: python -m guide_ex.recipe catalogue"
