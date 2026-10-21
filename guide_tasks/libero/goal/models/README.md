# Models GUIDE does not distribute

`wine_rack` and `wine_bottle` (LIBERO-Goal) are TurboSquid purchases, and TurboSquid's licence does not allow giving away the model
files. `../../assets/wine_rack/` and `../../assets/wine_bottle/` therefore hold only GUIDE's own definition of the asset (collision
boxes from LIBERO's MJCF, mass, material bindings). The mesh data and the textures are not in the repository: the committed
`.usda` files reference this folder, and the folders `models/<asset>/` are git-ignored.

Fill them once, from your own copy of LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO, commit 8f1084e):

    guide-bench assets libero --out <repo>/guide_tasks/libero --libero <LIBERO>/libero/libero

That converts LIBERO's original MJCF and meshes (`guide_core/scripts/spike_libero_convert.py`, needs `mujoco_usd_converter`) and puts
`GeometryLibrary.usdc`, `MaterialsLibrary.usdc` and `Textures/` into `models/<asset>/`. Until then registering a Goal task fails with
`scene recipe asset files are missing` and the names of the missing files, and `guide-bench translate` refuses the Goal suite.

The per-asset switch is `MODEL_EXTERNAL` in `guide_bench/readers/libero.py` (asset -> the suite whose `models/` folder holds it; an
asset used by several suites names one suite's folder).
