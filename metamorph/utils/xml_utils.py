import re
import os

# Attributes not supported by mujoco_py (introduced in MuJoCo 2.2+)
_UNSUPPORTED_ATTRS = ['actuatorfrcrange', 'actuatorfrclimited']

# Floor/environment elements to inject when the XML has no floor
_FLOOR_GEOM  = '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" condim="3" friction="1 0.005 0.0001"/>'
_FLOOR_LIGHT = '    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>'
_FLOOR_ASSETS = """\
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane"
              texuniform="true" texrepeat="5 5" reflectance="0.2"/>"""


def strip_unsupported_attrs(xml_path: str) -> str:
    """
    Produces a mujoco_py-compatible version of xml_path by:
      1. Stripping MuJoCo 2.2+ attributes (actuatorfrcrange etc.)
      2. Injecting a floor plane + lighting if the XML has no floor geom

    Writes the result next to the original as *_stripped.xml so that
    relative meshdir paths still resolve correctly.
    Returns the absolute path to the stripped file.
    """
    with open(xml_path, 'r') as f:
        content = f.read()

    # ── 1. Strip unsupported attributes ─────────────────────────────────────
    for attr in _UNSUPPORTED_ATTRS:
        content = re.sub(rf'\s+{attr}="[^"]*"', '', content)
        content = re.sub(rf"\s+{attr}='[^']*'", '', content)

    # ── 2. Inject floor if missing ───────────────────────────────────────────
    # Check for any plane geom (floor) already present
    has_floor = bool(re.search(r'type=["\']plane["\']', content))

    if not has_floor:
        # 2a. Add texture + material to <asset> block
        #     If no <asset> block exists, create one before <worldbody>
        if '</asset>' in content:
            content = content.replace(
                '</asset>',
                f'{_FLOOR_ASSETS}\n  </asset>'
            )
        else:
            content = content.replace(
                '<worldbody>',
                f'<asset>\n{_FLOOR_ASSETS}\n  </asset>\n\n  <worldbody>'
            )

        # 2b. Add light + floor geom as first children of <worldbody>
        content = content.replace(
            '<worldbody>',
            f'<worldbody>\n{_FLOOR_LIGHT}\n{_FLOOR_GEOM}'
        )

    # ── 3. Write stripped file next to original ──────────────────────────────
    out_path = xml_path.replace('.xml', '_stripped.xml')
    with open(out_path, 'w') as f:
        f.write(content)

    return out_path