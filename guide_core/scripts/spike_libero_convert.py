"""Spike S1: LIBERO-Spatial MJCF -> USD. Throwaway (branch spike/libero-assets).

  ~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py --selftest
  ~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path.home() / "libero_assets/LIBERO/libero/libero/assets"
MJCF = Path.home() / "libero_assets/mjcf"
USD = Path.home() / "libero_assets/usd"
CONVERTER = Path.home() / "ros2_ws/.venv/bin/mujoco_usd_converter"
OUT = Path(__file__).resolve().parents[2] / "docs/design/spike_libero_convert.json"
ASSETS = {
    "akita_black_bowl": "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
    "plate": "stable_scanned_objects/plate/plate.xml",
    "glazed_rim_porcelain_ramekin": "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml",
    "cookies": "stable_hope_objects/cookies/cookies.xml",
    "wooden_cabinet": "articulated_objects/wooden_cabinet.xml",
    "flat_stove": "articulated_objects/flat_stove.xml",
}


def read_msh(path):
    raw = Path(path).read_bytes()
    nv, nn, nt, nf = np.frombuffer(raw[:16], "<i4")
    if len(raw) != 16 + 12 * nv + 12 * nn + 8 * nt + 12 * nf:
        raise ValueError(f"{path}: size {len(raw)} does not match header {nv, nn, nt, nf}")
    o = 16
    def take(n, dtype, width):
        nonlocal o
        a = np.frombuffer(raw[o:o + 4 * n * width], dtype).reshape(n, width)
        o += 4 * n * width
        return a
    return {"vertices": take(nv, "<f4", 3), "normals": take(nn, "<f4", 3),
            "texcoords": take(nt, "<f4", 2), "faces": take(nf, "<i4", 3)}


def msh_to_obj(src, dst):
    m = read_msh(src)
    v, n, t, f = m["vertices"], m["normals"], m["texcoords"], m["faces"] + 1
    lines = [f"v {a:.7g} {b:.7g} {c:.7g}" for a, b, c in v]
    lines += [f"vn {a:.7g} {b:.7g} {c:.7g}" for a, b, c in n]
    lines += [f"vt {a:.7g} {1 - b:.7g}" for a, b in t]  # MuJoCo flips v for OBJ only
    def corner(i):  # i, i/i, i//i or i/i/i
        if not (len(t) or len(n)):
            return str(i)
        return f"{i}/{i if len(t) else ''}" + (f"/{i}" if len(n) else "")
    lines += ["f " + " ".join(corner(i) for i in face) for face in f]
    Path(dst).write_text("\n".join(lines) + "\n")


def patch(name):
    """Copy the asset into MJCF/<name>/, .msh -> .obj; return (patched xml, #msh converted)."""
    xml = SRC / ASSETS[name]
    dst = MJCF / name
    shutil.rmtree(dst, ignore_errors=True)
    if xml.parent.name == "articulated_objects":  # xml + sibling part folder
        dst.mkdir(parents=True)
        shutil.copy(xml, dst / xml.name)
        shutil.copytree(xml.parent / name, dst / name)
    else:
        shutil.copytree(xml.parent, dst)
    n = 0
    for msh in dst.rglob("*.msh"):
        msh_to_obj(msh, msh.with_suffix(".obj"))
        n += 1
    out = dst / xml.name
    text = re.sub(r'(file="[^"]*)\.msh"', r'\1.obj"', out.read_text())
    text = re.sub(r'<mujoco model="[^"]*"', f'<mujoco model="{name}"', text)  # usd is named after the model
    # flat_stove reuses a texture name; mujoco 3.x rejects it. Rename repeats, materials follow the latest def.
    seen, cur = {}, {}
    def tex(m):
        k = m[1]; seen[k] = seen.get(k, 0) + 1
        cur[k] = k if seen[k] == 1 else f"{k}_{seen[k]}"
        return m[0].replace(f'name="{k}"', f'name="{cur[k]}"')
    text = re.sub(r'<texture [^>]*name="([^"]+)"[^>]*>', tex, text)
    text = re.sub(r'(<material [^>]*texture=")([^"]+)"', lambda m: f'{m[1]}{cur.get(m[2], m[2])}"', text)
    out.write_text(text)
    return out, n


def convert(name):
    res = {"ok": False, "error": None, "usd": str(USD / name / f"{name}.usda"),
           "msh_converted": 0, "bodies": 0, "joints": [], "meshes": 0}
    try:
        shutil.rmtree(USD / name, ignore_errors=True)
        xml, res["msh_converted"] = patch(name)
        p = subprocess.run([str(CONVERTER), str(xml), str(USD / name), "--no-physics-scene"],
                           capture_output=True, text=True, timeout=600)
        if p.returncode:
            res["error"] = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
            return res
        import usdex.core  # noqa: F401  (needed for standalone pxr)
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(res["usd"])
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                res["bodies"] += 1
            if prim.IsA(UsdPhysics.Joint):
                res["joints"].append({"name": prim.GetName(), "type": prim.GetTypeName()})
            if prim.IsA(UsdGeom.Mesh):
                res["meshes"] += 1
        res["ok"] = True
    except Exception as e:  # spike: record and move on
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def main():
    results = {n: convert(n) for n in ASSETS}
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for n, r in results.items():
        print(n, "ok" if r["ok"] else f"FAIL {r['error']}")


def _selftest():
    with tempfile.TemporaryDirectory() as d:
        msh = Path(d) / "tri.msh"
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4")
        tex = np.array([[0, 0], [1, 0], [0, 0.25]], "<f4")
        faces = np.array([[0, 1, 2]], "<i4")
        msh.write_bytes(np.array([3, 0, 3, 1], "<i4").tobytes() + verts.tobytes() + tex.tobytes() + faces.tobytes())
        m = read_msh(msh)
        assert m["vertices"].shape == (3, 3) and m["faces"].tolist() == [[0, 1, 2]]
        obj = Path(d) / "tri.obj"
        msh_to_obj(msh, obj)
        text = obj.read_text()
        assert "vt 0 0.75" in text, text          # v flipped
        assert "f 1/1 2/2 3/3" in text, text      # 1-based, texcoords, no normals
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
