"""Genome -> MuJoCo MJCF XML.

Conventions: x is forward, z is up. Every actuated joint gets a position
servo (or a velocity servo for wheels) with a torque limit of
`genome.max_torque`, and joint names carry a type prefix that the CPG
controller uses to share parameters (e.g. all `knee_*` joints share one
amplitude/offset).
"""
from __future__ import annotations

import random

from .genome import Genome

FOOT_R = 0.035   # point-foot / capsule radius
WHEEL_R = 0.07
LIMB_R = 0.035


def _hinge(name: str, axis: str, lo: float, hi: float, stiffness: float = 0.0, damping: float = 1.0) -> str:
    extra = f' stiffness="{stiffness:.3f}" springref="0"' if stiffness > 0 else ""
    return (f'<joint name="{name}" type="hinge" axis="{axis}" range="{lo:.1f} {hi:.1f}" '
            f'damping="{damping}" armature="0.01"{extra}/>')


def _leg(g: Genome, tag: str, x: float, y: float) -> tuple[str, list[str]]:
    """One leg rooted at (x, y) under the pelvis. Returns (xml, actuated joint names)."""
    joints = [f"hiproll_{tag}", f"hippitch_{tag}", f"knee_{tag}"]
    knee_range = (0, 150) if g.knee_dir == "forward" else (-150, 0)
    xml = f'''
      <body name="thigh_{tag}" pos="{x:.3f} {y:.3f} 0">
        {_hinge(f"hiproll_{tag}", "1 0 0", -30, 30)}
        {_hinge(f"hippitch_{tag}", "0 1 0", -120, 60)}
        <geom type="capsule" fromto="0 0 0 0 0 {-g.thigh_len:.3f}" size="{LIMB_R}" rgba=".55 .6 .7 1"/>
        <body name="shin_{tag}" pos="0 0 {-g.thigh_len:.3f}">
          {_hinge(f"knee_{tag}", "0 1 0", *knee_range, stiffness=g.knee_spring)}
          <geom type="capsule" fromto="0 0 0 0 0 {-g.shin_len:.3f}" size="{LIMB_R * 0.85:.3f}" rgba=".45 .5 .6 1"/>'''
    end = f"{-g.shin_len:.3f}"
    if g.foot == "point":
        xml += f'\n          <geom name="foot_{tag}" type="sphere" pos="0 0 {end}" size="{FOOT_R}" friction="1.2" rgba=".2 .2 .2 1"/>'
    elif g.foot == "wheel":
        joints.append(f"wheel_{tag}")
        xml += f'''
          <body name="wheel_{tag}" pos="0 0 {end}">
            <joint name="wheel_{tag}" type="hinge" axis="0 1 0" damping="0.05"/>
            <geom name="foot_{tag}" type="cylinder" fromto="0 -0.025 0 0 0.025 0" size="{WHEEL_R}" friction="1.5" rgba=".15 .15 .15 1"/>
          </body>'''
    else:  # flat foot with ankle (and optional toe)
        joints.append(f"ankle_{tag}")
        heel, toe_x = -0.3 * g.foot_len, 0.7 * g.foot_len
        main_len = toe_x - heel if not g.toe else 0.75 * (toe_x - heel)
        xml += f'''
          <body name="foot_{tag}" pos="0 0 {end}">
            {_hinge(f"ankle_{tag}", "0 1 0", -45, 45, stiffness=g.ankle_spring)}
            <geom name="foot_{tag}" type="box" pos="{heel + main_len / 2:.3f} 0 -0.02" size="{main_len / 2:.3f} 0.045 0.02" friction="1.0" rgba=".2 .2 .2 1"/>'''
        if g.toe:
            joints.append(f"toe_{tag}")
            toe_len = (toe_x - heel) - main_len
            xml += f'''
            <body name="toe_{tag}" pos="{heel + main_len:.3f} 0 -0.02">
              {_hinge(f"toe_{tag}", "0 1 0", -10, 50)}
              <geom type="box" pos="{toe_len / 2:.3f} 0 0" size="{toe_len / 2:.3f} 0.045 0.02" friction="1.0" rgba=".25 .25 .25 1"/>
            </body>'''
        xml += "\n          </body>"
    xml += "\n        </body>\n      </body>"
    return xml, joints


def _arm(g: Genome, side: str, y: float, z: float) -> tuple[str, list[str]]:
    upper, fore = 0.5 * g.arm_len, 0.5 * g.arm_len
    xml = f'''
        <body name="uparm_{side}" pos="0 {y:.3f} {z:.3f}">
          {_hinge(f"shoulder_{side}", "0 1 0", -180, 60)}
          <geom type="capsule" fromto="0 0 0 0 0 {-upper:.3f}" size="0.03" rgba=".6 .65 .75 1"/>
          <body name="forearm_{side}" pos="0 0 {-upper:.3f}">
            {_hinge(f"elbow_{side}", "0 1 0", -150, 0)}
            <geom type="capsule" fromto="0 0 0 0 0 {-fore:.3f}" size="0.025" rgba=".6 .65 .75 1"/>
          </body>
        </body>'''
    return xml, [f"shoulder_{side}", f"elbow_{side}"]


def standing_height(g: Genome) -> float:
    foot = {"point": FOOT_R, "wheel": WHEEL_R, "flat": 0.04}[g.foot]
    return g.leg_len() + foot + 0.06  # pelvis half-height + small drop


def _terrain(kind: str, seed: int) -> str:
    if kind == "flat":
        return ""
    rng = random.Random(seed)
    boxes = []
    for i in range(40):  # scattered low obstacles / rubble over the first 6 m
        x, y = rng.uniform(0.6, 6.0), rng.uniform(-0.8, 0.8)
        h = rng.uniform(0.01, 0.05)
        boxes.append(f'<geom name="rubble{i}" type="box" pos="{x:.2f} {y:.2f} {h:.3f}" size="{rng.uniform(.05, .2):.2f} {rng.uniform(.05, .2):.2f} {h:.3f}" '
                     f'euler="0 0 {rng.uniform(0, 90):.0f}" contype="0" conaffinity="1" rgba=".5 .45 .4 1"/>')
    return "\n    ".join(boxes)


def build_xml(g: Genome, terrain: str = "flat", seed: int = 0) -> tuple[str, list[str]]:
    """Return (MJCF string, ordered list of actuated joint names)."""
    actuated: list[str] = []
    z0 = standing_height(g)
    hw = g.pelvis_width / 2
    if g.leg_pairs == 1:
        hips = [("l", 0.0, hw), ("r", 0.0, -hw)]
    else:
        fx = g.pelvis_len / 2 - 0.05
        hips = [("fl", fx, hw), ("fr", fx, -hw), ("rl", -fx, hw), ("rr", -fx, -hw)]

    legs = ""
    for tag, x, y in hips:
        xml, j = _leg(g, tag, x, y)
        legs += xml
        actuated += j

    upper = ""
    if g.chest_height > 0:
        chest_x = g.pelvis_len / 2 - 0.08 if g.leg_pairs == 2 else 0.0
        spine = ""
        if g.spine:
            spine = _hinge("spine", "0 1 0", -40, 40, stiffness=g.spine_stiffness)
            actuated.append("spine")
        arms = ""
        if g.arms:
            for side, y in (("l", hw + 0.06), ("r", -hw - 0.06)):
                xml, j = _arm(g, side, y, g.chest_height * 0.9)
                arms += xml
                actuated += j
        upper = f'''
      <body name="chest" pos="{chest_x:.3f} 0 0.05">
        {spine}
        <geom type="box" pos="0 0 {g.chest_height / 2:.3f}" size="0.09 {hw:.3f} {g.chest_height / 2:.3f}" density="350" rgba=".8 .5 .3 1"/>
        <geom type="sphere" pos="0 0 {g.chest_height + 0.1:.3f}" size="0.09" density="400" rgba=".9 .9 .9 1"/>{arms}
      </body>'''

    tail = ""
    if g.tail_len > 0:
        tail = f'''
      <body name="tail" pos="{-g.pelvis_len / 2:.3f} 0 0">
        {_hinge("tail", "0 1 0", -60, 60)}
        <geom type="capsule" fromto="0 0 0 {-g.tail_len:.3f} 0 0.05" size="0.025" density="600" rgba=".7 .6 .5 1"/>
      </body>'''
        actuated.append("tail")

    act_xml = []
    for j in actuated:
        if j.startswith("wheel"):
            act_xml.append(f'<velocity name="{j}" joint="{j}" kv="4" ctrlrange="-25 25" forcelimited="true" forcerange="{-g.max_torque / 4:.1f} {g.max_torque / 4:.1f}"/>')
        else:
            act_xml.append(f'<position name="{j}" joint="{j}" kp="{2.0 * g.max_torque:.1f}" kv="{0.05 * g.max_torque:.2f}" inheritrange="1" '
                           f'forcelimited="true" forcerange="{-g.max_torque:.1f} {g.max_torque:.1f}"/>')

    xml = f'''<mujoco model="hrethink">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <!-- robot geoms collide only with the environment, not with each other -->
  <default><geom contype="1" conaffinity="0" condim="3" density="800"/></default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1" contype="0" conaffinity="1" rgba=".85 .85 .8 1" friction="1.0"/>
    {_terrain(terrain, seed)}
    <body name="pelvis" pos="0 0 {z0:.3f}">
      <freejoint name="root"/>
      <geom name="pelvis" type="box" size="{g.pelvis_len / 2:.3f} {hw + 0.02:.3f} 0.05" density="500" rgba=".8 .5 .3 1"/>
      <site name="imu" pos="0 0 0"/>{legs}{upper}{tail}
    </body>
  </worldbody>
  <actuator>
    {chr(10).join("    " + a for a in act_xml).strip()}
  </actuator>
</mujoco>'''
    return xml, actuated
