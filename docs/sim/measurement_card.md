# SO101 measurement card (WOWROBO rig → MuJoCo twin)

About 10 minutes with calipers and a ruler. It fixes the twin's reach error: near rest it's within
~1 cm, but at reach the sim claws sit 3–11 cm high (docs/sim/kincal_status.md). Values go in
`~/sim_ref/measurements.json` (template: `measurements.template.json`), in **mm**, per arm. The
**right (blue) arm matters most**; do the left too if time allows.

**Stock** = the value in the sim's model (MuJoCo Menagerie SO101). If a reading is off by more than
~5 mm (or 5°), that's the finding: re-measure once and add a photo.

"Axis centre" = the centre screw of that servo's output horn (the round disc the next link is bolted
to). Measure centre-to-centre on the same side of the arm.

## A. Static lengths (any pose, arms can rest)

| # | From → to (physical reference points) | Maps to (kincal / model) | Stock |
|---|---|---|---|
| A1 | shoulder-lift axis centre → elbow axis centre | `upper_arm_dx` = A1 − stock | **116.0** |
| A2 | elbow axis centre → wrist-flex axis centre | `forearm_dx` = A2 − stock | **135.0** |
| A3 | wrist-flex axis centre → **claw tip**, gripper CLOSED. Tip = outermost point of the **fixed** jaw (the one that doesn't move) | `claw_dx` ≈ A3 − stock | **159.7** |
| A4 | tabletop → shoulder-lift axis centre (vertical height) | `base_dz` = A4 − stock | **116.6** |
| A5 | how each base is attached: `on_table` (bottom plate sits on the tabletop) or `clamped_edge` (plate/clamp below the table surface). If clamped: tabletop → bottom of the base plate, mm (negative = below) | base placement (the model's plate extends 44 mm below the base origin) | on_table, 0 |
| A6 | right pan axis → left pan axis (centres of the round horns on top of each base), horizontal | `BASE_Y` = A6 / 2 | **350.0** |
| A7 | each pan axis → table's back edge, horizontal. + if the edge is in FRONT of the pan axis, − if behind | table placement in the sim | (sim: table spans the bases) |

## B. Heights at two commanded poses (the 1080 session commands them; both arms)

Commanded (normalized): **pan 0, wrist_roll 0, gripper 5** on both arms.
- **P0 "all-zero":** lift 0, elbow 0, wrist_flex 0.
- **P1:** lift 0, elbow 20, wrist_flex 0.

Save `/state` at each pose too. P1's claw tip is predicted 147 mm up; even if the real arm reaches
11 cm lower, it stays clear. Approach slowly anyway.

| # | Height above the TABLETOP, vertical | P0 stock | P1 stock |
|---|---|---|---|
| B1 | elbow axis centre | **229** | **229** |
| B2 | wrist-flex axis centre | **234** | **190** |
| B3 | claw tip (as in A3) | **246** | **147** |
| B4 | *(optional)* claw tip's horizontal distance in front of its own pan axis | **353** | **342** |

These pin the joint zero conventions and offsets. Together with A1–A3, the fitter separates:
- **B1:** the shoulder-lift offset;
- **B2:** the elbow offset;
- **B3/B4:** the wrist-flex offset.

The question to settle is lift ≈ +32° / elbow ≈ −14°, which is what the "homing" hypothesis
predicts for the right arm, vs. 0 / 0.

## C. Wrist camera plate (right arm; left if quick)

Reference: the gripper's **long axis** = the wrist-roll axis (the centreline the claws extend along).

| # | Measurement | Stock |
|---|---|---|
| C1 | perpendicular distance: gripper long axis → centre of the camera **lens** | **60.0** |
| C2 | along the long axis: wrist-flex axis centre → the lens's position (projected onto the axis) | **101** |
| C3 | angle between the plate's face and the gripper long axis (0° = face parallel to the claws, 90° = face looking straight down the claws) | **58.5°** |
| C4 | photo from straight in front of the claws, gripper closed, showing which side the plate sticks out | — |

## Optional: AprilTags (only if a printer is handy)

Print `~/sim_ref/apriltags_36h11_30mm.pdf` at 100% (no "fit to page"). Each black square must
measure **30.0 mm**; check with the calipers.

Placement:
- **Right arm:** ID0 upper arm (outer face), ID1 forearm (outer face), ID2 wrist-flex servo body,
  ID3 fixed jaw (flat face near the tip).
- **Table corners:** ID4–ID7, flat on the tabletop, roughly 30 cm apart.

Then the 1080 captures overhead + both wrist frames at P0, P1 and rest. Tags give each link's full
pose and also fix the wrist-camera placement, which the photos alone couldn't recover.

## Where the numbers go

Once `~/sim_ref/measurements.json` exists:
- the fitter adds "point height at a measured pose" residuals;
- lengths are fixed from section A, and the camera plate from section C;
- joint offsets / zero convention are fit from section B, with the table contacts kept as a check;
- then the wrist-camera and side-view comparisons are re-validated.
