"""Candy scene for the MuJoCo twin: a wide, shallow bowl of assorted candy, a plate to put candy on,
and a human hand that reaches in to take one.

Every candy type exists once in the model. reset_candy() drops a random subset into the bowl and
parks the rest on a hidden shelf under the table (MuJoCo can't add bodies without recompiling).
The bowl, plate and hand are mocap bodies, so episodes can move them without recompiling.

Candy frame: local x = long axis, z = up when lying naturally, origin = geometric centre.
Colors are solid and matte, so a sim-trained policy doesn't learn foil glare.
"""
import math

import mujoco
import numpy as np

_G = mujoco.mjtGeom

BOWL = {"pos": (0.20, 0.0), "floor_r": 0.09, "wall": 0.035, "flare_deg": 40.0, "fill_r": 0.062}
BOWL_FLOOR_TOP = 0.006  # bowl floor surface height (the bowl mocap sits at z = 0)
PLATE = {"r": 0.05, "top": 0.004}
PARK_Z = -0.30  # hidden shelf under the tabletop
SHELF = {"pos": (0.25, 0.0, PARK_Z - 0.01), "half": (0.30, 0.45, 0.01)}
MOCAP_PARK = {"candy_bowl": (0.25, 0.30, PARK_Z + 0.05), "plate": (0.25, -0.30, PARK_Z + 0.05),
              "human_hand": (0.10, 0.0, PARK_Z + 0.08)}

WRAPPERS = [[0.85, 0.12, 0.12, 1], [0.15, 0.35, 0.85, 1], [0.1, 0.65, 0.25, 1], [0.95, 0.8, 0.1, 1],
            [0.55, 0.2, 0.7, 1], [0.95, 0.5, 0.1, 1], [0.95, 0.95, 0.92, 1], [0.95, 0.45, 0.65, 1]]
SKIN = [[0.96, 0.8, 0.67, 1], [0.87, 0.67, 0.5, 1], [0.66, 0.46, 0.32, 1], [0.42, 0.28, 0.2, 1]]
PLATE_COLORS = [[0.95, 0.95, 0.93, 1], [0.2, 0.6, 0.9, 1], [0.3, 0.75, 0.35, 1], [0.95, 0.85, 0.2, 1]]

CONTACT = dict(condim=6, friction=[1.3, 0.01, 0.002])  # grippy, like the plush toys: no spin-out in the jaws
VISUAL = dict(contype=0, conaffinity=0, mass=0.0001)

# name -> hw (half-width across the jaws), words (for instructions), rest_z (centre height when lying on a flat surface),
# half_len (along local x), grasp (local pick point),
# handover = (x where the giving arm holds it, x where the receiving arm grabs it, half-width there); None = too short
# for two claws-down grippers side by side
CANDIES = {
    "chocolate_bar": dict(hw=0.016, words="chocolate bar", rest_z=0.0065, half_len=0.050, grasp=(0, 0, 0), handover=(-0.026, 0.030, 0.016)),
    "mini_bar": dict(hw=0.013, words="candy bar", rest_z=0.0075, half_len=0.030, grasp=(0, 0, 0), handover=None),
    "wrapped_candy": dict(hw=0.011, words="wrapped candy", rest_z=0.011, half_len=0.038, grasp=(0, 0, 0), handover=None),
    "lollipop": dict(hw=0.017, words="lollipop", rest_z=0.017, half_len=0.053, grasp=(0.035, 0, 0), handover=(0.035, -0.036, 0.0035)),
    "licorice": dict(hw=0.007, words="licorice", rest_z=0.007, half_len=0.056, grasp=(0, 0, 0), handover=(-0.028, 0.034, 0.007)),
    "candy_corn": dict(hw=0.011, words="candy corn", rest_z=0.0065, half_len=0.013, grasp=(0, 0, 0), handover=None),
    "peanut_cup": dict(hw=0.021, words="peanut butter cup", rest_z=0.006, half_len=0.021, grasp=(0, 0, 0), handover=None),
    "gumball": dict(hw=0.012, words="gumball", rest_z=0.012, half_len=0.012, grasp=(0, 0, 0), handover=None),
    "candy_box": dict(hw=0.017, words="candy box", rest_z=0.009, half_len=0.025, grasp=(0, 0, 0), handover=None),
    "gummy_bear": dict(hw=0.009, words="gummy bear", rest_z=0.0075, half_len=0.016, grasp=(0, 0, 0), handover=None),
    "pumpkin_candy": dict(hw=0.017, words="pumpkin candy", rest_z=0.012, half_len=0.017, grasp=(0, 0, 0), handover=None),
}
ROUND = {"gumball", "peanut_cup", "pumpkin_candy"}  # no long axis: any wrist roll grips them
# geoms whose color is re-drawn from WRAPPERS each reset (name suffix)
RECOLOR = ("wrapper", "label", "ball", "box", "stick_head")

_X = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]  # rotates a capsule/cylinder's z axis onto x


def _yquat(angle):
    return [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]


def _zquat(angle):
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out.tolist()


def _frustum(spec, name, r0, r1, z0, z1, n=16):
    verts = []
    for r, z in ((r0, z0), (r1, z1)):
        for k in range(n):
            a = 2 * math.pi * k / n
            verts += [r * math.cos(a), r * math.sin(a), z]
    spec.add_mesh(name=name, uservert=verts)


def _prism(spec, name, outline, z0, z1):
    """Convex prism from a 2-D outline (x, y) between heights z0 and z1."""
    verts = [c for x, y in outline for z in (z0, z1) for c in (x, y, z)]
    spec.add_mesh(name=name, uservert=verts)


def _ring(body, prefix, r_in, z0, length, flare_deg, thick, n, rgba):
    """Flared wall: n tilted boxes whose bottoms sit on a circle of radius r_in at height z0."""
    fl = math.radians(flare_deg)
    rc, zc = r_in + 0.5 * length * math.sin(fl), z0 + 0.5 * length * math.cos(fl)
    half_w = rc * math.sin(math.pi / n) + thick
    for k in range(n):
        a = 2 * math.pi * k / n
        body.add_geom(name=f"{prefix}{k}", type=_G.mjGEOM_BOX, size=[thick, half_w, length / 2],
                      pos=[rc * math.cos(a), rc * math.sin(a), zc], quat=_qmul(_zquat(a), _yquat(fl)), rgba=rgba)


def _add_candy(spec, wb, name: str, slot: int):
    body = wb.add_body(name=name, pos=[*_park_xy(slot), PARK_Z + 0.03])
    body.add_freejoint(name=f"{name}_free")
    brown, cream = [0.36, 0.2, 0.1, 1], [0.97, 0.95, 0.88, 1]
    g = dict(CONTACT)
    if name == "chocolate_bar":
        body.add_geom(name=f"{name}_wrapper", type=_G.mjGEOM_BOX, size=[0.050, 0.016, 0.0065], mass=0.05, rgba=brown, **g)
        body.add_geom(name=f"{name}_label", type=_G.mjGEOM_BOX, size=[0.014, 0.0163, 0.0068], rgba=WRAPPERS[0], **VISUAL)
    elif name == "mini_bar":
        body.add_geom(name=f"{name}_wrapper", type=_G.mjGEOM_BOX, size=[0.030, 0.013, 0.0075], mass=0.025, rgba=WRAPPERS[1], **g)
    elif name == "wrapped_candy":
        body.add_geom(name=f"{name}_wrapper", type=_G.mjGEOM_CAPSULE, size=[0.011, 0.012, 0], quat=_X, mass=0.015,
                      rgba=WRAPPERS[3], **g)
        for s in (-1, 1):  # twisted ends
            body.add_geom(name=f"{name}_end{s + 1}", type=_G.mjGEOM_ELLIPSOID, size=[0.008, 0.010, 0.003],
                          pos=[s * 0.030, 0, 0], mass=0.002, rgba=[0.9, 0.9, 0.9, 1], **g)
    elif name == "lollipop":
        body.add_geom(name=f"{name}_stick_head", type=_G.mjGEOM_SPHERE, size=[0.017, 0, 0], pos=[0.035, 0, 0],
                      mass=0.015, rgba=WRAPPERS[0], **g)
        body.add_geom(name=f"{name}_stick", type=_G.mjGEOM_CAPSULE, size=[0.003, 0.035, 0], pos=[-0.015, 0, -0.013],
                      quat=_X, mass=0.004, rgba=cream, **g)
    elif name == "licorice":
        body.add_geom(name=f"{name}_rope", type=_G.mjGEOM_CAPSULE, size=[0.007, 0.049, 0], quat=_X, mass=0.014,
                      rgba=[0.8, 0.08, 0.1, 1], **g)
    elif name == "candy_corn":  # rounded triangle, lying flat, in three bands: yellow base, orange, white tip
        bands = [((-0.013, 0.011), (-0.004, 0.0075), [0.98, 0.8, 0.1, 1]),
                 ((-0.004, 0.0075), (0.005, 0.004), [0.97, 0.5, 0.08, 1]),
                 ((0.005, 0.004), (0.013, 0.0008), [0.98, 0.97, 0.92, 1])]
        for i, ((x0, w0), (x1, w1), rgba) in enumerate(bands):
            _prism(spec, f"{name}_band{i}_mesh", [(x0, -w0), (x0, w0), (x1, -w1), (x1, w1)], -0.0065, 0.0065)
            body.add_geom(name=f"{name}_band{i}", type=_G.mjGEOM_MESH, meshname=f"{name}_band{i}_mesh",
                          mass=0.004, rgba=rgba, **g)
    elif name == "peanut_cup":
        _frustum(spec, f"{name}_mesh", 0.0195, 0.0205, -0.006, 0.006)  # near-straight wall: a steep taper wedges it out of the jaws
        body.add_geom(name=f"{name}_wrapper", type=_G.mjGEOM_MESH, meshname=f"{name}_mesh", mass=0.02,
                      rgba=[0.95, 0.5, 0.1, 1], **g)
        body.add_geom(name=f"{name}_top", type=_G.mjGEOM_CYLINDER, size=[0.018, 0.0008, 0], pos=[0, 0, 0.0055],
                      rgba=brown, **VISUAL)
    elif name == "gumball":
        body.add_geom(name=f"{name}_ball", type=_G.mjGEOM_SPHERE, size=[0.012, 0, 0], mass=0.01, rgba=WRAPPERS[4], **g)
    elif name == "candy_box":
        body.add_geom(name=f"{name}_box", type=_G.mjGEOM_BOX, size=[0.025, 0.017, 0.009], mass=0.03, rgba=WRAPPERS[7], **g)
        body.add_geom(name=f"{name}_label", type=_G.mjGEOM_BOX, size=[0.012, 0.0172, 0.0092], rgba=cream, **VISUAL)
    elif name == "gummy_bear":  # lying on its back
        gb = dict(g, rgba=[0.9, 0.15, 0.2, 0.9])
        body.add_geom(name=f"{name}_body", type=_G.mjGEOM_ELLIPSOID, size=[0.010, 0.008, 0.0075], pos=[-0.003, 0, 0],
                      mass=0.008, **gb)
        body.add_geom(name=f"{name}_head", type=_G.mjGEOM_SPHERE, size=[0.0065, 0, 0], pos=[0.011, 0, 0], mass=0.003, **gb)
        for i, (x, y, r) in enumerate([(0.016, 0.005, 0.0025), (0.016, -0.005, 0.0025), (0.002, 0.009, 0.003),
                                       (0.002, -0.009, 0.003), (-0.011, 0.007, 0.0032), (-0.011, -0.007, 0.0032)]):
            body.add_geom(name=f"{name}_limb{i}", type=_G.mjGEOM_SPHERE, size=[r, 0, 0], pos=[x, y, -0.001],
                          mass=0.0005, **gb)
    elif name == "pumpkin_candy":
        body.add_geom(name=f"{name}_body", type=_G.mjGEOM_ELLIPSOID, size=[0.017, 0.017, 0.012], mass=0.012,
                      rgba=[0.97, 0.47, 0.06, 1], **g)
        body.add_geom(name=f"{name}_stem", type=_G.mjGEOM_CYLINDER, size=[0.003, 0.003, 0], pos=[0, 0, 0.013],
                      rgba=[0.2, 0.5, 0.15, 1], **VISUAL)
    else:
        raise KeyError(name)


def _park_xy(slot: int) -> tuple[float, float]:
    return 0.05 + 0.08 * (slot % 4), -0.30 + 0.12 * (slot // 4)


def _add_hand(wb):
    """Palm-up, slightly cupped child's hand. Origin = centre of the palm surface; fingers point
    along local -x, the forearm leaves along +x and up (toward the person)."""
    # a dynamic hand welded to a mocap target: a teleported mocap body has no velocity, so friction could not
    # carry a candy in its palm; the welded body really moves and takes the candy with it
    wb.add_body(name="hand_target", mocap=True, pos=list(MOCAP_PARK["human_hand"]))
    hand = wb.add_body(name="human_hand", pos=list(MOCAP_PARK["human_hand"]), gravcomp=1.0)
    hand.add_freejoint(name="hand_free")
    skin = dict(rgba=SKIN[0], condim=4, friction=[1.2, 0.01, 0.001])
    hand.add_geom(name="hand_palm", type=_G.mjGEOM_BOX, size=[0.034, 0.032, 0.008], pos=[0, 0, -0.008], **skin)
    curl = math.radians(28)
    for i, (y, length) in enumerate([(-0.024, 0.036), (-0.008, 0.044), (0.008, 0.046), (0.024, 0.040)]):
        c = [-0.034 - 0.5 * length * math.cos(curl), y, 0.5 * length * math.sin(curl) - 0.003]
        hand.add_geom(name=f"hand_finger{i}", type=_G.mjGEOM_CAPSULE, size=[0.0072, length / 2, 0], pos=c,
                      quat=_yquat(curl - math.pi / 2), **skin)
    # thumb on +y (a right hand; reset mirrors it for a left hand by flipping the yaw side it comes from)
    hand.add_geom(name="hand_thumb", type=_G.mjGEOM_CAPSULE, size=[0.0085, 0.019, 0], pos=[-0.006, 0.043, 0.006],
                  quat=_qmul(_zquat(math.radians(-60)), _yquat(math.radians(-70))), **skin)
    hand.add_geom(name="hand_heel", type=_G.mjGEOM_ELLIPSOID, size=[0.012, 0.030, 0.010], pos=[0.028, 0, 0.003], **skin)
    hand.add_geom(name="hand_edge", type=_G.mjGEOM_ELLIPSOID, size=[0.030, 0.008, 0.008], pos=[-0.004, -0.036, 0.003],
                  **skin)  # the little-finger side of the palm: with the thumb and fingers it makes a cup
    up = math.radians(35)
    hand.add_geom(name="hand_forearm", type=_G.mjGEOM_CAPSULE, size=[0.021, 0.10, 0],
                  pos=[0.045 + 0.10 * math.cos(up), 0, -0.012 + 0.10 * math.sin(up)],
                  quat=_yquat(math.pi / 2 - up), **skin)


def add_candy_scene(spec: mujoco.MjSpec):
    wb = spec.worldbody
    _add_hand(wb)
    weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name="hand_weld",
                             name1="human_hand", name2="hand_target", solref=[0.005, 1.0])
    weld.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * (len(weld.data) - 11)  # hand frame = target frame
    shelf_pos, shelf_half = SHELF["pos"], SHELF["half"]
    wb.add_geom(name="candy_shelf", type=_G.mjGEOM_BOX, size=list(shelf_half), pos=list(shelf_pos), rgba=[0, 0, 0, 0])
    purple = [0.45, 0.2, 0.6, 1]
    bowl = wb.add_body(name="candy_bowl", mocap=True, pos=list(MOCAP_PARK["candy_bowl"]))
    bowl.add_geom(name="bowl_floor", type=_G.mjGEOM_CYLINDER, size=[BOWL["floor_r"] + 0.004, 0.006, 0], pos=[0, 0, 0.0],
                  rgba=purple)
    _ring(bowl, "bowl_wall", BOWL["floor_r"], 0.004, BOWL["wall"], BOWL["flare_deg"], 0.0025, 24, purple)
    plate = wb.add_body(name="plate", mocap=True, pos=list(MOCAP_PARK["plate"]))
    plate.add_geom(name="plate_base", type=_G.mjGEOM_CYLINDER, size=[PLATE["r"] + 0.002, PLATE["top"] / 2, 0],
                   pos=[0, 0, PLATE["top"] / 2], rgba=PLATE_COLORS[0])
    _ring(plate, "plate_rim", PLATE["r"], PLATE["top"], 0.014, 55.0, 0.002, 20, PLATE_COLORS[0])
    for slot, name in enumerate(CANDIES):
        _add_candy(spec, wb, name, slot)


# ---- runtime ----------------------------------------------------------------------------
def _mocap_id(model, body: str) -> int:
    return int(model.body_mocapid[model.body(body).id])


def set_mocap(model, data, body: str, pos, yaw: float = 0.0):
    i = _mocap_id(model, body)
    data.mocap_pos[i] = pos
    data.mocap_quat[i] = _zquat(yaw)


def set_hand(model, data, pos, yaw: float = 0.0, teleport: bool = False):
    """Move the hand's target; teleport=True also puts the hand itself there (resets, not mid-episode)."""
    set_mocap(model, data, "hand_target", pos, yaw)
    if teleport:
        _set_free(model, data, "hand", pos, yaw)


def _set_free(model, data, name: str, pos, yaw: float):
    adr = model.joint(f"{name}_free").qposadr[0]
    data.qpos[adr:adr + 7] = [*pos, *_zquat(yaw)]
    dadr = model.joint(f"{name}_free").dofadr[0]
    data.qvel[dadr:dadr + 6] = 0.0


def _recolor(model, rng, group: str, colors):
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith(group):
            model.geom_rgba[g] = colors


def reset_candy(model, data, rng: np.random.Generator, n_candies: int | None = None, randomize: bool = True,
                include: str | None = None) -> dict:
    """Bowl at its spot (±2 cm when randomized), a random subset of candies dropped into it, the rest
    parked; plate and hand parked (episodes place them). Call mj_forward after. Returns the layout."""
    bx, by = BOWL["pos"]
    if randomize:
        bx, by = bx + rng.uniform(-0.02, 0.02), by + rng.uniform(-0.02, 0.02)
    set_mocap(model, data, "candy_bowl", [bx, by, 0.0])
    set_mocap(model, data, "plate", MOCAP_PARK["plate"])
    set_hand(model, data, MOCAP_PARK["human_hand"], teleport=True)
    names = list(CANDIES)
    n = n_candies or int(rng.integers(3, 7))
    chosen = [str(c) for c in rng.choice(names, size=min(n, len(names)), replace=False)] if randomize else names[:n]
    if include and include not in chosen:
        chosen[0] = include
    placed = []
    for name in chosen:  # rejection-sample a spot in the bowl, clear of the ones already placed
        info = CANDIES[name]
        for _ in range(60):
            r = BOWL["fill_r"] * math.sqrt(rng.uniform()) if randomize else 0.02 * len(placed)
            a = rng.uniform(0, 2 * math.pi) if randomize else 1.3 * len(placed)
            p = np.array([bx + r * math.cos(a), by + r * math.sin(a)])
            if all(np.linalg.norm(p - q) > 0.8 * (info["half_len"] + hl) + 0.015 for q, hl in placed):
                break
        yaw = float(rng.uniform(-math.pi, math.pi)) if randomize else 0.0
        _set_free(model, data, name, [p[0], p[1], 0.006 + info["rest_z"] + 0.004], yaw)
        placed.append((p, info["half_len"]))
    for slot, name in enumerate(names):
        if name not in chosen:
            _set_free(model, data, name, [*_park_xy(slot), PARK_Z + 0.03], 0.0)
    if randomize:  # solid, high-contrast wrapper colors
        for name in CANDIES:
            for part in RECOLOR:
                _recolor(model, rng, f"{name}_{part}", WRAPPERS[int(rng.integers(len(WRAPPERS)))])
        _recolor(model, rng, "plate_", PLATE_COLORS[int(rng.integers(len(PLATE_COLORS)))])
        _recolor(model, rng, "hand_", SKIN[int(rng.integers(len(SKIN)))])
    return {"bowl": [bx, by], "candies": chosen}


def candy_pose(model, data, name: str) -> tuple[np.ndarray, np.ndarray]:
    """World position of the candy's origin and its long axis (unit, world frame)."""
    b = model.body(name).id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3)[:, 0].copy()


def candy_point(model, data, name: str, local) -> np.ndarray:
    b = model.body(name).id
    return data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.asarray(local, float)


def in_bowl(model, data, name: str) -> bool:
    bx, by = data.mocap_pos[_mocap_id(model, "candy_bowl")][:2]
    p = data.xpos[model.body(name).id]
    return bool(np.hypot(p[0] - bx, p[1] - by) < BOWL["floor_r"] + 0.02 and p[2] < 0.05)


def on_plate(model, data, name: str) -> bool:
    c = data.mocap_pos[_mocap_id(model, "plate")]
    p = data.xpos[model.body(name).id]
    return bool(np.hypot(p[0] - c[0], p[1] - c[1]) < PLATE["r"] and p[2] < c[2] + 0.04)


def hand_frame(model, data) -> tuple[np.ndarray, np.ndarray]:
    b = model.body("human_hand").id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def in_hand(model, data, name: str) -> bool:
    """Candy resting in the palm: within the palm/finger footprint and at most 4 cm above the palm."""
    pos, R = hand_frame(model, data)
    local = R.T @ (data.xpos[model.body(name).id] - pos)
    return bool(-0.07 < local[0] < 0.045 and abs(local[1]) < 0.045 and -0.005 < local[2] < 0.04)
