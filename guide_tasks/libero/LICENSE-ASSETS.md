# LIBERO assets in GUIDE: provenance and licences

The files under `assets/` are **not part of GUIDE and are not covered by GUIDE's GPL-3.0 licence.** They are under licences different from GUIDE's.

- Each asset keeps the licence of its source. The licences are listed below and in the `LICENSE` file in each asset's folder.
- The assets ship beside GUIDE as an aggregate, converted for Isaac Sim, so that the LIBERO tasks under `spatial/` run without a LIBERO checkout.
- Check an asset's licence before you redistribute it or use it commercially: `cookies` is non-commercial and share-alike, and the terms of `wooden_cabinet` and `flat_stove` are unverified.
- This file is a record of provenance, not legal advice.

The stage (`stage/`) and the task configurations (`spatial/`) are GUIDE's own work under GUIDE's licence.

The wood of the cabinet (carcass and drawers) and the metal of its handles are NVIDIA Base materials (`NVIDIA/Materials/Base/Wood/Walnut.mdl`, `Metals/Steel_Stainless.mdl`), loaded by URL from NVIDIA's asset server when the scene is opened. They are under the NVIDIA Omniverse License, not GUIDE's GPL-3.0, and are not part of this repository. The other materials are Isaac Sim's own `OmniPBR` and `OmniPBR_ClearCoat` (BSD-3-Clause), referenced by name, and the dome light's HDRI (`NVIDIA/Assets/Skies/Indoor/studio_small_04_4k.hdr`) is likewise loaded by URL. The stage's table texture, `stage/textures/martin_novak_wood_table.png`, is LIBERO's own file and has the same unverified terms as LIBERO's other assets.

## Source

- **LIBERO:** Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", NeurIPS 2023
  Datasets and Benchmarks, https://github.com/Lifelong-Robot-Learning/LIBERO at commit 8f1084e, directory
  `libero/libero/assets/`.
  - LIBERO's code is MIT-licensed (text below).
  - Its README lists "Datasets: CC BY 4.0"; the scope of that line (demos only, or also assets) is unclear.
  - The repository has no separate licence or attribution file for its 3D assets.
- **Changes for GUIDE:**
  - MuJoCo MJCF/OBJ/MSH converted to USD with `mujoco_usd_converter` (`guide_core/scripts/spike_libero_convert.py`);
  - physics overrides authored by `guide_bench` (`assets.py`): the welds of movable objects disabled, and collision-only geometry hidden from rendering, and the visual meshes rebound to MDL materials.

## Per asset

| Asset | Origin | Licence | Notes |
|---|---|---|---|
| `plate` | Google Scanned Objects, Google LLC | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/legalcode | The original model files carry Google LLC's CC BY 4.0 header. |
| `akita_black_bowl` | Google Scanned Objects (Google LLC) via LIBERO, re-modelled; origin not proven | CC BY 4.0 if the Google Scanned Objects origin holds (unverified), https://creativecommons.org/licenses/by/4.0/legalcode | Attribute both Google Scanned Objects and LIBERO. |
| `glazed_rim_porcelain_ramekin` | as above | as above | as above |
| `cookies` | NVIDIA HOPE dataset (Tyree et al., IROS 2022), https://github.com/swtyree/hope-dataset | CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode | NonCommercial and ShareAlike: not usable commercially, adaptations stay CC BY-NC-SA 4.0; likely incompatible with GPL terms, so kept as a separately licensed file. BOP lists CC BY-SA 4.0; the stricter is assumed. |
| `wooden_cabinet` | LIBERO (Lifelong Robot Learning); origin of meshes and textures unknown | No licence file for LIBERO's assets; LIBERO's code is MIT and its README lists "Datasets: CC BY 4.0", whose scope (demos only, or also assets) is unclear | Origin and redistribution terms unverified. |
| `flat_stove` | as above | as above | as above |

Attribution:
- "Google Scanned Objects, Google LLC, CC BY 4.0" (https://arxiv.org/abs/2204.11918);
- "HOPE, NVIDIA Corporation, CC BY-NC-SA 4.0";
- "LIBERO, Liu et al. 2023, Lifelong Robot Learning".

## LIBERO's licence (code)

    MIT License

    Copyright (c) 2023 Lifelong Robot Learning

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
