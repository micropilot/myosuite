"""Generate Stage C keyboard MORPHOLOGY variants for the AAAI transfer study.

All variants are TACTILE (spring-loaded slider keys) and differ only in trivial,
physically-meaningful parameters of make_keyboard.generate():

  base   : the shipped board (pitch 0.019, travel 0.004, stiffness 80)  [unchanged]
  tight  : key pitch 0.017 (denser keyboard)
  stiff  : travel 0.006 + stiffness 140 + damping 1.5 (deeper, harder keys)
  ortho  : alt LAYOUT -- remove the QWERTY row stagger (ortholinear columns)

For each variant it writes:
  qwerty_keyboard_<v>.xml    (the keyboard include fragment, all 61 keys)
  myohand_keyboard_<v>.xml   (top-level scene = myohand_keyboard.xml with the
                              keyboard include swapped to the variant fragment)

The env loads a variant by overriding `model_path` -> myohand_keyboard_<v>.xml.
Run from this directory (so the relative <include>s resolve):
  python make_stagec_variants.py
"""
import os
import make_keyboard as mk

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TOP = os.path.join(HERE, "myohand_keyboard.xml")

# (tag, generate-kwargs, row_stagger-override-or-None)
VARIANTS = [
    ("tight", dict(pitch=0.017), None),
    ("stiff", dict(travel=0.006, stiffness=140.0, damping=1.5), None),
    ("ortho", dict(), [0.0, 0.0, 0.0, 0.0, 3.5]),
]


def main():
    top = open(BASE_TOP).read()
    assert '<include file="qwerty_keyboard.xml"/>' in top, \
        "base top-level does not include qwerty_keyboard.xml"
    orig_stagger = list(mk.ROW_STAGGER)
    for tag, kw, stagger in VARIANTS:
        mk.ROW_STAGGER = stagger if stagger is not None else list(orig_stagger)
        frag = mk.generate(with_sensors=True, model_name=f"qwerty_keyboard_{tag}", **kw)
        frag_name = f"qwerty_keyboard_{tag}.xml"
        with open(os.path.join(HERE, frag_name), "w") as f:
            f.write(frag)
        top_v = top.replace('<include file="qwerty_keyboard.xml"/>',
                            f'<include file="{frag_name}"/>')
        top_name = f"myohand_keyboard_{tag}.xml"
        with open(os.path.join(HERE, top_name), "w") as f:
            f.write(top_v)
        n_keys = frag.count('<body name="key_')
        print(f"wrote {frag_name} + {top_name}  ({n_keys} keys, kw={kw}, "
              f"stagger={mk.ROW_STAGGER})")
    mk.ROW_STAGGER = orig_stagger


if __name__ == "__main__":
    main()
