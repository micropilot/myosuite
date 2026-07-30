"""Generate a QWERTY keyboard as a MuJoCo `mujocoinclude` fragment.

Each key is an independent body on a prismatic (slide) joint with a return
spring, a box geom, a top site, and a `touch` sensor. A "keypress" is detected
either from the key travel (slide qpos past a threshold) or from the touch
sensor contact force. The generated fragment is included by
`myohand_keyboard.xml`.

Run:  python make_keyboard.py            # writes qwerty_keyboard.xml
Layout coordinates:  x = across keyboard (columns), y = rows (near->far),
z = up (keys travel down in -z when pressed).
"""
from __future__ import annotations

import argparse
import os

# Standard staggered QWERTY. Each entry: (label, width_in_units).
# `label` is the emg2qwerty-style key name; width in key-pitch units.
ROWS: list[list[tuple[str, float]]] = [
    [("`", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1),
     ("7", 1), ("8", 1), ("9", 1), ("0", 1), ("-", 1), ("=", 1),
     ("Key.backspace", 2)],
    [("Key.tab", 1.5), ("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1),
     ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1), ("[", 1), ("]", 1),
     ("\\", 1.5)],
    [("Key.caps_lock", 1.75), ("a", 1), ("s", 1), ("d", 1), ("f", 1),
     ("g", 1), ("h", 1), ("j", 1), ("k", 1), ("l", 1), (";", 1), ("'", 1),
     ("Key.enter", 2.25)],
    [("Key.shift", 2.25), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1),
     ("n", 1), ("m", 1), (",", 1), (".", 1), ("/", 1), ("Key.shift_r", 2.75)],
    [("Key.space", 8.0)],
]

# Left edge offset (in units) of each row, giving the classic stagger.
ROW_STAGGER = [0.0, 0.0, 0.25, 0.75, 3.5]


def sanitize(label: str) -> str:
    """Map a key label to a MuJoCo-safe unique name."""
    special = {
        "`": "backtick", "-": "minus", "=": "equal", "[": "lbracket",
        "]": "rbracket", "\\": "backslash", ";": "semicolon",
        "'": "quote", ",": "comma", ".": "period", "/": "slash",
    }
    if label in special:
        return special[label]
    if label.startswith("Key."):
        return label[4:]  # Key.space -> space
    return label  # letters / digits


def generate(
    pitch: float = 0.019,     # 19.05 mm real key pitch
    gap: float = 0.002,       # gap between keycaps
    travel: float = 0.004,    # key travel depth (m)
    cap_height: float = 0.004,
    base_pos=(-0.325, -0.600, 1.380),  # keyboard body world origin (under hand)
    stiffness: float = 80.0,   # key return spring
    damping: float = 1.0,
    keep: "set | None" = None,  # if set, only emit keys whose sanitized name is in it
    with_sensors: bool = True,  # emit touch sensors (MJX/Warp may reject touch sensors)
    model_name: str = "qwerty_keyboard",
) -> str:
    """Generate the keyboard include fragment.

    `keep` restricts the emitted keys to a subset (useful for a SMALL GPU board)
    WHILE preserving every key's world coordinate -- the x/y accumulation still
    walks the full staggered layout, so a subset board is numerically identical
    to the full board for the keys it contains.  `with_sensors=False` drops the
    `<touch>` sensors (the MJX/MuJoCo-Warp backends can reject touch sensors)."""
    half = (pitch - gap) / 2.0
    bodies: list[str] = []
    sensors: list[str] = []
    keymap: list[str] = []  # comment listing label -> name

    y = 0.0
    # The hands are mounted on the +y side (forearms come from +y). For natural
    # typing the SPACE bar must be nearest the hands (largest y) and the number
    # row FAR (smallest y). ROWS is ordered number(0)..space(last), so row_y = r*
    # pitch puts the number row at the far edge and space nearest the typist.
    # (The previous (n_rows-1-r) ordering had it backwards -> hands reached over
    # the number row from behind.) The home row stays centred under the fingers.
    n_rows = len(ROWS)
    for r, row in enumerate(ROWS):
        row_y = r * pitch
        x = ROW_STAGGER[r] * pitch
        for label, width in row:
            w_half = (width * pitch - gap) / 2.0
            cx = x + w_half  # center of this (possibly wide) key
            name = sanitize(label)
            x += width * pitch  # advance BEFORE any skip so coords stay identical
            if keep is not None and name not in keep:
                continue
            # position relative to the keyboard body (which is placed at base_pos)
            gpos = f"{cx:.5f} {row_y:.5f} {cap_height/2:.5f}"
            # gravcomp="1" keeps the key from sagging under gravity so it rests
            # at the top of its travel until a finger presses it.
            bodies.append(f"""      <body name="key_{name}" pos="{gpos}" gravcomp="1">
        <joint name="key_{name}_slide" type="slide" axis="0 0 -1" range="0 {travel:.4f}"
               stiffness="{stiffness}" damping="{damping}" springref="0" armature="0.0005"/>
        <geom name="key_{name}_geom" type="box" size="{w_half:.5f} {half:.5f} {cap_height/2:.5f}"
              rgba="0.15 0.15 0.17 1" mass="0.005" friction="1 0.005 0.0001"/>
        <site name="key_{name}_site" type="box" size="{w_half:.5f} {half:.5f} {cap_height/2:.5f}"
              rgba="0.3 0.6 0.9 0.15"/>
      </body>""")
            sensors.append(f'    <touch name="key_{name}_touch" site="key_{name}_site"/>')
            keymap.append(f"{label} -> key_{name}")

    body_xml = "\n".join(bodies)
    sensor_block = ""
    if with_sensors:
        sensor_xml = "\n".join(sensors)
        sensor_block = f"""
  <sensor>
{sensor_xml}
  </sensor>"""
    keymap_comment = "\n".join("       " + k for k in keymap)

    return f"""<mujocoinclude model="{model_name}">
<!-- AUTO-GENERATED by make_keyboard.py. Do not edit by hand.
     pitch={pitch} gap={gap} travel={travel} cap_height={cap_height}
     Key label -> body name:
{keymap_comment}
-->
  <worldbody>
    <body name="keyboard" pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}">
      <!-- Decorative base plate (contype/conaffinity=0 -> no collision, so it
           never blocks key travel or spawns key-plate contacts). Sits below the
           key's lowest travel point. -->
      <geom name="keyboard_plate" type="box" contype="0" conaffinity="0"
            size="0.16 0.06 0.004" pos="0.14 0.038 -0.012"
            rgba="0.08 0.08 0.09 1" mass="1.0"/>
{body_xml}
    </body>
  </worldbody>{sensor_block}
</mujocoinclude>
"""


def generate_from_dataset(
    csv_path: str,
    travel: float = 0.004,
    cap_height: float = 0.004,
    key_half: float = 0.0085,   # half key-cap size (~17mm cap on 19mm pitch)
    base_pos=(-0.309, -0.609, 1.380),
    stiffness: float = 80.0,
    damping: float = 1.0,
) -> str:
    """Build a keyboard whose key positions are taken VERBATIM from the Aalto
    'How We Type' dataset (keyboard_flat_coordinates.csv), so the env layout is
    identical to the mocap keyboard used for imitation. Dataset frame:
    x=across, y=height, z=rows (Escape=(0,0,0)). We map dataset (x, z) into the
    keyboard body plane (x, y) preserving the exact relative layout, oriented
    like the generated keyboard (number row far, space near)."""
    # dataset key name -> our sanitized name (letters/digits share names)
    alias = {"space": "space", "comma": "comma", "period": "period",
             "minus": "minus", "equal": "equal"}
    coords = {}
    with open(csv_path) as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            name, x, y, z = p[0], *(float(v) for v in p[1:])
            if name in "abcdefghijklmnopqrstuvwxyz" or name in "0123456789":
                coords[name] = (x, z)
            elif name in alias:
                coords[alias[name]] = (x, z)
    xs = [c[0] for c in coords.values()]
    zs = [c[1] for c in coords.values()]
    x0, z0 = min(xs), min(zs)
    bodies, sensors, keymap = [], [], []
    for name, (x, z) in sorted(coords.items()):
        cx = (x - x0) + 0.02           # across
        cy = (z - z0) + 0.01           # rows: numbers far (large), space near
        gpos = f"{cx:.5f} {cy:.5f} {cap_height/2:.5f}"
        bodies.append(f"""      <body name="key_{name}" pos="{gpos}" gravcomp="1">
        <joint name="key_{name}_slide" type="slide" axis="0 0 -1" range="0 {travel:.4f}"
               stiffness="{stiffness}" damping="{damping}" springref="0" armature="0.0005"/>
        <geom name="key_{name}_geom" type="box" size="{key_half:.5f} {key_half:.5f} {cap_height/2:.5f}"
              rgba="0.15 0.15 0.17 1" mass="0.005" friction="1 0.005 0.0001"/>
        <site name="key_{name}_site" type="box" size="{key_half:.5f} {key_half:.5f} {cap_height/2:.5f}"
              rgba="0.3 0.6 0.9 0.15"/>
      </body>""")
        sensors.append(f'    <touch name="key_{name}_touch" site="key_{name}_site"/>')
        keymap.append(f"{name} -> key_{name}")
    plate_cx = ((max(xs) - x0) + 0.02) / 2 + 0.01
    plate_cy = ((max(zs) - z0) + 0.01) / 2
    return f"""<mujocoinclude model="qwerty_keyboard_dataset">
<!-- AUTO-GENERATED by make_keyboard.py --from-dataset. Do not edit by hand.
     Key positions taken from the Aalto How-We-Type dataset for imitation.
     Key name -> body name:
{chr(10).join('       ' + k for k in keymap)}
-->
  <worldbody>
    <body name="keyboard" pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}">
      <geom name="keyboard_plate" type="box" contype="0" conaffinity="0"
            size="0.16 0.06 0.004" pos="{plate_cx:.3f} {plate_cy:.3f} -0.012"
            rgba="0.08 0.08 0.09 1" mass="1.0"/>
{chr(10).join(bodies)}
    </body>
  </worldbody>
  <sensor>
{chr(10).join(sensors)}
  </sensor>
</mujocoinclude>
"""


# Small right-hand board used by the GPU/MJX-Warp keyboard task (milestone 1).
# Right/center region reachable by a single right MyoHand; coordinates identical
# to the full board.  Sanitized key names (see sanitize()).
MINI_RIGHT_HAND_KEYS = {
    "6", "7", "8", "9", "0",
    "y", "u", "i", "o", "p",
    "h", "j", "k", "l", "semicolon",
    "n", "m", "comma", "period",
    "space",
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dataset", default=None,
                    help="path to keyboard_flat_coordinates.csv (dataset-matched layout)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "qwerty_keyboard.xml"))
    ap.add_argument("--mini", action="store_true",
                    help="emit only the right-hand key subset, no touch sensors "
                         "(small board for the MJX/Warp GPU task)")
    ap.add_argument("--no-sensors", action="store_true",
                    help="omit touch sensors (MJX/Warp can reject them)")
    args = ap.parse_args()
    if args.from_dataset:
        xml = generate_from_dataset(args.from_dataset)
        n_keys = xml.count("<body name=\"key_")
    elif args.mini:
        xml = generate(keep=MINI_RIGHT_HAND_KEYS, with_sensors=False,
                       model_name="qwerty_keyboard_mini")
        n_keys = len(MINI_RIGHT_HAND_KEYS)
    else:
        xml = generate(with_sensors=not args.no_sensors)
        n_keys = sum(len(r) for r in ROWS)
    with open(args.out, "w") as f:
        f.write(xml)
    print(f"wrote {args.out} ({n_keys} keys)")
