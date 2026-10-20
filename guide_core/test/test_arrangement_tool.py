"""python -m guide_core.tools.arrangement: a recorded episode's Replicator draws as an arrangement."""
import json

import yaml

from guide_core.tools import arrangement as tool


def dataset(tmp_path, *lines):
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    (meta / "guide_episodes.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return tmp_path / "ds"


def test_an_episodes_single_draws_become_the_arrangement(tmp_path):
    ds = dataset(
        tmp_path,
        {"episode_index": 3, "values": {"replicator/plate_position": [[0.0, 0.0, 0.0]]}},
        {"episode_index": 3, "values": {"color": "red", "replicator/plate_position": [[0.05, 0.2, 0.0033]],
                                         "replicator/cabinet_top_state": [-0.12]}},  # the last line wins
    )
    out = tmp_path / "demo_3.yaml"
    assert tool.main([str(ds), "3", str(out)]) == 0
    assert yaml.safe_load(out.read_text()) == {"plate_position": [0.05, 0.2, 0.0033], "cabinet_top_state": -0.12}


def test_a_distribution_drawn_for_several_prims_is_refused(tmp_path, capsys):
    ds = dataset(tmp_path, {"episode_index": 0, "values": {"replicator/blocks_position": [[0, 0, 0], [1, 1, 1]]}})
    assert tool.main([str(ds), "0", str(tmp_path / "x.yaml")]) == 2
    assert "blocks_position (2 samples)" in capsys.readouterr().err
    assert not (tmp_path / "x.yaml").exists()


def test_a_missing_episode_or_sidecar_is_refused(tmp_path, capsys):
    ds = dataset(tmp_path, {"episode_index": 0, "values": {"replicator/p": [[0, 0, 0]]}})
    assert tool.main([str(ds), "7", str(tmp_path / "x.yaml")]) == 2
    assert tool.main([str(tmp_path / "nowhere"), "0", str(tmp_path / "x.yaml")]) == 2
    err = capsys.readouterr().err
    assert "no episode 7" in err and "guide_episodes.jsonl" in err
