"""Generate a LEFT MyoHand by mirroring the right MyoHand MJCF.

MyoSuite ships only a right MyoHand. This script reflects the right-hand model
across the sagittal (x=0) plane to produce an anatomically-correct left hand,
renaming every entity with an ``_L`` suffix so the two hands can coexist in one
scene.

Mirror math (reflection D = diag(-1, 1, 1), applied as the frame similarity
L' = D L D so stored rotations stay proper/right-handed, with chirality moved
into the mesh geometry via negative-x mesh scale):

  pos            (x, y, z)                 -> (-x, y, z)
  quat           (w, x, y, z)              -> (w, x, -y, -z)
  euler          -> converted to quat, mirrored, emitted as quat
  joint axis     (ax, ay, az)              -> (-ax, ay, az)
  joint range    (lo, hi)                  -> (-hi, -lo)     (coordinate flips)
  geom fromto    (x1,y1,z1, x2,y2,z2)      -> negate x1, x2
  fullinertia    (ixx,iyy,izz,ixy,ixz,iyz) -> negate ixy, ixz
  mesh scale     (sx, sy, sz)              -> (-sx, sy, sz)  (mirror chirality)

Reads  myo_sim/hand/assets/{myohand_body.xml, myohand_assets.xml}
Writes this dir's       {myohand_body_left.xml, myohand_assets_left.xml}

Run:  python mirror_hand.py
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "..", "simhive", "myo_sim", "hand", "assets")
SUFFIX = "_L"

# Attributes that DEFINE a named entity we want to rename.
NAME_ATTRS = ("name",)
# Reference attributes -> whether the referenced kind is renamed by us.
# (class/childclass/material are shared with the right hand -> NOT renamed.)
REF_ATTRS = ("mesh", "site", "geom", "sidesite", "tendon", "joint",
             "body1", "body2", "geom1", "geom2", "body")


def _floats(s):
    return [float(x) for x in s.split()]


def _fmt(vals):
    return " ".join(repr(v) if isinstance(v, float) else str(v) for v in vals)


def euler_to_quat(rx, ry, rz):
    """MuJoCo default eulerseq='xyz' (intrinsic X, then Y, then Z), radians."""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    # q = qx * qy * qz  (intrinsic xyz)
    # qx=(cx,sx,0,0) qy=(cy,0,sy,0) qz=(cz,0,0,sz)
    # qx*qy:
    w1 = cx * cy
    x1 = sx * cy
    y1 = cx * sy
    z1 = sx * sy  # (sx*0 - ... ) compute properly below
    # do full Hamilton products to avoid mistakes
    def mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    q = mul(mul((cx, sx, 0, 0), (cy, 0, sy, 0)), (cz, 0, 0, sz))
    return list(q)


def mirror_quat(q):
    w, x, y, z = q
    return [w, x, -y, -z]


def transform_geometry(elem):
    """Apply the sagittal mirror to one element's numeric attributes in place."""
    tag = elem.tag

    if "pos" in elem.attrib:
        p = _floats(elem.attrib["pos"])
        p[0] = -p[0]
        elem.attrib["pos"] = _fmt(p)

    # orientation: prefer quat; convert euler->quat then mirror
    if "quat" in elem.attrib:
        elem.attrib["quat"] = _fmt(mirror_quat(_floats(elem.attrib["quat"])))
    elif "euler" in elem.attrib:
        q = euler_to_quat(*_floats(elem.attrib["euler"]))
        del elem.attrib["euler"]
        elem.attrib["quat"] = _fmt(mirror_quat(q))

    if tag == "joint":
        if "axis" in elem.attrib:
            a = _floats(elem.attrib["axis"])
            a[0] = -a[0]
            elem.attrib["axis"] = _fmt(a)
        if "range" in elem.attrib:
            lo, hi = _floats(elem.attrib["range"])
            elem.attrib["range"] = _fmt([-hi, -lo])

    if tag == "geom" and "fromto" in elem.attrib:
        f = _floats(elem.attrib["fromto"])
        f[0] = -f[0]
        f[3] = -f[3]
        elem.attrib["fromto"] = _fmt(f)

    if tag == "inertial" and "fullinertia" in elem.attrib:
        I = _floats(elem.attrib["fullinertia"])  # ixx iyy izz ixy ixz iyz
        I[3] = -I[3]  # ixy
        I[4] = -I[4]  # ixz
        elem.attrib["fullinertia"] = _fmt(I)

    if tag == "mesh" and "scale" in elem.attrib:
        s = _floats(elem.attrib["scale"])
        s[0] = -s[0]
        elem.attrib["scale"] = _fmt(s)


def collect_names(root):
    """All entity names we intend to rename (bodies/joints/geoms/sites/meshes/
    tendons/actuators). Materials & default classes are shared -> excluded."""
    names = set()
    rename_tags = {"body", "joint", "geom", "site", "mesh", "spatial",
                   "muscle", "general", "motor", "position", "tendon",
                   "camera", "light"}
    for e in root.iter():
        if e.tag in rename_tags and "name" in e.attrib:
            names.add(e.attrib["name"])
    return names


def rename_refs(root, names):
    for e in root.iter():
        # rename definitions
        if "name" in e.attrib and e.attrib["name"] in names:
            e.attrib["name"] = e.attrib["name"] + SUFFIX
        # rename references (only when they point at something we renamed)
        for attr in REF_ATTRS:
            if attr in e.attrib and e.attrib[attr] in names:
                e.attrib[attr] = e.attrib[attr] + SUFFIX


# Top-level sections in the assets file that are SHARED with the right hand
# (compiler/visual/default/etc.) or would duplicate (materials/textures). The
# left include must not redefine them, else MuJoCo errors on repeated classes.
STRIP_TOPLEVEL = {"compiler", "size", "option", "visual", "default"}
STRIP_ASSET_CHILDREN = {"material", "texture"}


def strip_shared(root):
    """For the assets file: keep only mirrored meshes/contact/tendon/actuator;
    drop shared scaffolding and duplicate materials/textures."""
    for tag in STRIP_TOPLEVEL:
        for e in root.findall(tag):
            root.remove(e)
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag in STRIP_ASSET_CHILDREN:
                asset.remove(child)


def drop_torso_geom(root):
    """Remove the full-body torso mesh geom (name 'body') so a bimanual scene
    shows just the two forearms/hands instead of two overlapping skeletons."""
    for parent in root.iter():
        for g in list(parent):
            if g.tag == "geom" and g.get("name") in ("body", "body" + SUFFIX):
                parent.remove(g)


def process(path_in, path_out, names_all, mirror=True, strip=False,
            drop_torso=False):
    tree = ET.parse(path_in)
    root = tree.getroot()
    if mirror:
        # 1) geometry mirror
        for e in root.iter():
            transform_geometry(e)
        # 2) rename definitions + references
        rename_refs(root, names_all)
    # 3) drop shared/duplicate sections (assets file only)
    if strip:
        strip_shared(root)
    if drop_torso:
        drop_torso_geom(root)
    ET.indent(tree, space="  ")
    tree.write(path_out, encoding="unicode", xml_declaration=False)
    return root


def main():
    body_in = os.path.join(SRC, "myohand_body.xml")
    assets_in = os.path.join(SRC, "myohand_assets.xml")

    # Collect the union of names across BOTH files so references resolve.
    names_all = set()
    for p in (body_in, assets_in):
        names_all |= collect_names(ET.parse(p).getroot())

    # Left hand: mirrored, torso geom dropped.
    body_out = os.path.join(HERE, "myohand_body_left.xml")
    assets_out = os.path.join(HERE, "myohand_assets_left.xml")
    process(body_in, body_out, names_all, mirror=True, drop_torso=True)
    process(assets_in, assets_out, names_all, mirror=True, strip=True)
    # Right hand: identity copy with torso dropped, for the bimanual scene
    # (keeps the submodule pristine; the unimanual env still uses the submodule).
    body_r_out = os.path.join(HERE, "myohand_body_right.xml")
    process(body_in, body_r_out, names_all, mirror=False, drop_torso=True)
    print("wrote", body_out)
    print("wrote", assets_out)
    print("wrote", body_r_out)
    print("mirrored %d named entities" % len(names_all))


if __name__ == "__main__":
    main()
