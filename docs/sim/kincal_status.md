# Sim ↔ real kinematic calibration: status (2026-09-26)

## What's active in the sim

- **Joint mapping:** unchanged. Every joint is zeroed at the middle of its calibrated range (lerobot DEGREES
  convention), with no offsets.
- **Overhead camera:** re-fit (`halloween_bot/sim/geometry.json`). The C922 was bumped during the rig
  visit. Fitting the camera alone to the lit overlay photos raises the blue-arm silhouette IoU from
  0.345 to 0.643 (Δpos ≈ 3–4 cm, pitch +7°).

## Evidence and what it says

| Evidence | Current mapping (A) | "homing" zero for lift/elbow + fitted residuals (B) |
|---|---|---|
| Real wrist-camera view directions, Sep-20 teleop episode (4 checks) | **3/4** | 1/4; puts the right wrist camera 3 cm *below* the table at one real frame |
| Overhead photo silhouettes, camera re-fit per model | **0.643** | 0.599 |
| Rest / grasp poses sit just above the table | ✓ (+1.1…+1.4 cm) | ✓ (≈0 cm) |
| 9 confirmed table contacts (`~/sim_ref/table_contacts_v3.json`) | ✗ claws **3–11 cm above** the table | **✓ within ±1.3 cm, holdout −0.9 / −0.1 cm** |
| No-contact descent paths stay above the table | ✓ | ✓ within 1.6 cm |

Neither model is consistent with all of the evidence. Models tried on the contacts:
- joint offsets only;
- a longer claw;
- a table plane (a 50° "tilt", which is unphysical);
- a base tilt;
- link lengths + offsets.

Each one either fails the contacts, or fits them only with geometry that the wrist-camera and
photo evidence reject. Under A the arm is not self-colliding at the contacts either: the gap is
≥ 2.4 cm.

**Most likely missing piece:** what actually touches at a "contact". The detector sees elbow tracking
error growing on one more commanded step. That is consistent with the claw tips pressing the table,
but also with some other part, or a surface that isn't at the bases' height.

## Next decisive measurement (needs the rig)

A side view at 2–3 confirmed contact poses (e.g. v3 rows 2, 3, 7). Two options:
- aim the orange arm's wrist camera at the blue claw from the side;
- a 10-second phone photo from the side.

Either shows directly what touches and how far the claw tip is from the table. With that:
- if the claw tips really touch, the zero convention (B) is right, and the wrist-camera model or the
  Sep-20 data needs a second look;
- otherwise, the contact rows get reinterpreted and A stands.

## Tooling (tested, inactive until a model is chosen)

- `halloween_bot/sim/kincal.py` — joint fit of link-length deltas, joint offsets, base tilt and the
  overhead camera. It fits against confirmed contacts, free descent paths, anchor poses, and photo
  silhouettes (Powell), with a holdout.
- `Calibration(zero={"shoulder_lift": "homing", ...})` — the per-joint zero convention, computed per
  arm from its own calibration file. Set in `geometry.json` `"zero"`.
- `apply_geometry()` — link lengths, claw length, base height/tilt, overhead camera pose.
