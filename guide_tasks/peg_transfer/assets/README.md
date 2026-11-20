# Peg-transfer assets

- `dvrk/`, `dvrk_model/`, `surrol/`: sources and licences, see [LICENSE-ASSETS.md](LICENSE-ASSETS.md).
- `peg_transfer.usd`: the whole scene (SUJ cart, ECM with stereo cameras and scope light, PSM1, PSM2, peg board,
  6 blocks) as one flattened, self-contained usdc. Generated; do not edit by hand.

## Regenerating `peg_transfer.usd`

The scene is built by davinci_sim (`~/ros2_ws/src/davinci_sim`, not a git repo) and exported here:

```
cd ~/ros2_ws/src/davinci_sim && ~/isaacsim/python.sh build_scene.py   # -> scene/davinci_scene.usda, scene_info.json
cd <guide> && ~/ros2_ws/.venv/bin/python -I guide_tasks/peg_transfer/tools/export_scene.py
```

`--src <davinci_sim>/scene` and `--out <file>` override the defaults. The export needs only usd-core (no Isaac);
PhysX schemas are authored by name. It:

- asserts each arm `root_joint` is anchored to the world (empty body0, at the SUJ-solved arm pose), so the file
  holds 4 articulations: the Cart (SUJ drives kept) and ECM, PSM1, PSM2;
- drops `/World/PhysicsScene` (GUIDE owns physics) and zeroes the joint velocities Isaac saved in the file;
- blocks: 2 g, CCD, 32 position iterations, max depenetration 0.05 m/s; block colliders and pegs: contact offset
  1 mm, rest offset 0; one physics material (friction 1.0, restitution 0) on blocks, pegs and jaws;
- jaw drives: 0.5 N m/rad, 0.01 N m s/rad, max 0.05 N m; arm articulations: 32 position iterations;
- grafted dvrk_model parallelogram links: 1e-4 kg (still mimic joints);
- adds the Xform `/World/PegTransfer/handover` 5 cm above the board centre;
- flattens with defaultPrim `/World` and asserts no external references or local asset paths remain. Materials
  left: `OmniPBR.mdl` (ships with Kit) and NVIDIA vMaterials `Stainless_Steel_Milled.mdl` by URL.

The tuning knobs (block mass, jaw drive, iterations, contact offset) are constants at the top of the script.
`test/test_assets.py` checks the result and skips when the file is absent.
