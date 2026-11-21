"""tools/path_stats.py on a synthetic path: an L whose corner is a free end the tip cuts by 5 mm."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("path_stats", Path(__file__).parents[1] / "tools" / "path_stats.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def test_free_corner_cut_deviation_and_contact_end():
    t = np.linspace(0, 1, 200)[:, None]
    leg1 = t * [0.05, 0.0, 0.0]
    arc = [0.05, 0.0, 0.0] + 0.005 * np.c_[-np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0]  # the cut's apex
    leg2 = [0.05, 0.0, 0.0] + t * [0.0, 0.05, 0.0]
    P = np.vstack([leg1[:150], arc, leg2[50:], np.repeat(leg2[-1:] - [0.0, 0.001, 0.0], 5, axis=0)])
    moves = [{"robot": "psm1", "position": [0.0, 0.0, 0.0]},
             {"robot": "psm1", "position": [0.05, 0.0, 0.0], "perturb": True, "precise_end": False},
             {"robot": "psm1", "position": [0.05, 0.05, 0.0], "speed": 0.05}]
    rec = {"moves": moves}
    idx = ps.match(P, ps.goals(rec, "psm1"))
    assert idx[0] == 0 and idx[1] == 150 and idx[2] == len(P) - 6  # the last goal first within TOL, then its minimum
    rows = ps.moves(rec, {"psm1": P, "psm2": P[:1]})
    first, second = rows
    assert first["kind"] == "perturbed" and abs(first["cut"] - 0.005) < 1e-6
    assert second["kind"] == "contact" and second["end"] < 1e-6
    s = ps.summary(rows)
    assert s["cut_mm"] == 5.0 and s["perturbed"] == 1
