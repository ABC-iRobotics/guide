from guide_ex.recipe.catalogue import _refs
from guide_ex.recipe.editor import DEFAULT_VALUES, YAML_RT, dump, fresh_registry, plain, rows, timeline
from guide_ex.recipe.loader import load
from guide_ex.recipe.registry import LIBRARY

RECIPES = [p for folder in LIBRARY for p in sorted(folder.glob("*.yaml"))]


def test_a_recipe_saves_as_it_was_read():
    for path in RECIPES:
        assert dump(YAML_RT.load(path.read_text())) == path.read_text(), path.name


def test_the_tree_shows_every_node_the_file_writes_once_in_the_file():
    """Its children, branches and fallbacks, and every override at the node it replaces."""
    registry = fresh_registry()
    for path in RECIPES:
        doc = YAML_RT.load(path.read_text())
        written = [row.spec["ref"] for row in rows(doc, registry) if row.slot and "ref" in row.spec]
        assert sorted(written) == sorted(_refs(plain(doc))), path.name


def test_the_timeline_of_pick_reads_each_pose_after_the_node_that_gives_it():
    registry = fresh_registry()
    root = load(LIBRARY[2] / "Pick.yaml", registry=registry, values=plain(YAML_RT.load(DEFAULT_VALUES)))
    out = []
    timeline(root, set(root.dynamic_map.values()), out)
    by_name = {name.strip(): (reads, lacking, gives) for _, name, reads, lacking, gives, _ in out}
    assert by_name["TransformCubeGraspPose"][2] == ["cube_grasp_pose"]
    assert "cube_grasp_pose" in by_name["MoveToCube"][0]
    assert not [name for _, name, _, lacking, _, _ in out if lacking]
