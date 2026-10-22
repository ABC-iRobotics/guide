from guide_ex.recipe.catalogue import _refs
from guide_ex.recipe.registry import LIBRARY, read_spec
from guide_ex.recipe.viewer import kids


def _walk(spec):
    return [spec.get("ref")] + [r for _, child in kids(spec) for r in _walk(child)]


def test_the_viewer_tree_shows_every_node_a_recipe_refs():
    for path in (p for folder in LIBRARY for p in folder.glob("*.yaml")):
        spec = read_spec(path)
        assert sorted(filter(None, _walk(spec))) == sorted(_refs(spec)), path.name
