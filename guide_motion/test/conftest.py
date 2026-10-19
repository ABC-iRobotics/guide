import json

import pytest
import yaml

from fr3_fixture import DATA, simulator_mass_properties, usd_poses
from guide_motion import description


@pytest.fixture(scope="session")
def fr3_dir(tmp_path_factory):
    urdf, xrdf, meta = description.build(
        (DATA / "fr3_exporter_raw.urdf").read_text(), yaml.safe_load((DATA / "fr3_editor_raw.xrdf").read_text()), usd_poses(), "fr3"
    )
    d = tmp_path_factory.mktemp("robots") / "fr3"
    d.mkdir()
    (d / "fr3.urdf").write_text(urdf)
    (d / "fr3.xrdf").write_text(yaml.safe_dump(xrdf, sort_keys=False))
    (d / "description.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="session")
def fr3_model(fr3_dir):
    from guide_motion.model import Model

    model = Model.load(fr3_dir)
    model.set_mass_properties(*simulator_mass_properties())
    return model
