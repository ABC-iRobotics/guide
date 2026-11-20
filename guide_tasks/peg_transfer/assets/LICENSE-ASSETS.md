# Peg-transfer assets in GUIDE: provenance and licences

The files under `assets/` are **not part of GUIDE and are not covered by GUIDE's GPL-3.0 licence.** Each keeps the licence of its source, listed below and in the `LICENSE` file of its folder. This file is a record of provenance, not legal advice.

| Folder | Origin | Licence | Changes |
|---|---|---|---|
| `dvrk/` | NVIDIA Isaac for Healthcare (i4h) v0.9.0, PSM and ECM USD, https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/v0.9.0/7cf86a1/Robots/dVRK/ | Apache-2.0, https://www.apache.org/licenses/LICENSE-2.0.txt | none |
| `dvrk_model/` | jhu-dvrk/dvrk_model at 90472a9, https://github.com/jhu-dvrk/dvrk_model | MIT, [LICENSE](dvrk_model/LICENSE) | link meshes embedded in `peg_transfer.usd` by the scene export |
| `surrol/` | med-air/SurRoL, Benchmark/state_based/surrol/assets (`peg_board_bimanual`, `block`), https://github.com/med-air/SurRoL | MIT, [LICENSE](surrol/LICENSE) | block mass 0.5 kg to grams, peg radius 1.5 mm, meshes converted to USD by davinci_sim `build_scene.py` |

The scene `peg_transfer.usd` combines the three sources and is GUIDE's own aggregate work; the embedded parts stay under the licences above.
