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

## Update: side views (`~/sim_ref/sideview/`, the orange wrist cam looking at the blue claw)

- **The real camera mount plate sticks out sideways**, like Menagerie's `camera_mount`. The 1080
  session read the plate as the part that touches the table.
- **The model can't reproduce that.** Menagerie's plate, included in the contact set, is never
  lowest by 4–11 cm under the current mapping. Even with ±90°/180° wrist-roll offsets the mean
  miss is ≥ 5 cm.
- **The sim can't reproduce the side view itself either.** Rendered from the orange wrist camera at
  the captured state, the image matches under no tested convention: range-middle, homing on
  lift/elbow or on all three, wrist_flex −40…+40°.

Conclusion: the real WOWROBO arms differ from stock SO101 in more than one way at once. Candidates are
the calibration zero convention, part geometry and camera-mount pose. Table contacts plus a few photos
can't separate those.

## Next decisive measurement (needs the rig) — superseded, see below

A side view at 2–3 confirmed contact poses (e.g. v3 rows 2, 3, 7). Two options:
- aim the orange arm's wrist camera at the blue claw from the side;
- a 10-second phone photo from the side.

Either shows directly what touches and how far the claw tip is from the table. With that:
- if the claw tips really touch, the zero convention (B) is right, and the wrist-camera model or the
  Sep-20 data needs a second look;
- otherwise, the contact rows get reinterpreted and A stands.

## Recommended path if sim accuracy at reach matters

Direct geometric calibration instead of inference:
- **Measure the real parts:** link lengths, claw length, and the camera plate offset and angle
  (≈10 min with calipers). Or put small AprilTags on the forearm, wrist and claw, then use the
  overhead camera (plus one extra view) to recover each link's pose at a handful of states.
- **Refit:** feed those into `kincal.py`; its Kin/apply_geometry/zero-convention plumbing is
  ready.

Until then the sim is accurate near the rest pose (≈1 cm) and increasingly optimistic about height
at reach (claws 3–11 cm higher than real at mid/far reach).

## Tooling (tested, inactive until a model is chosen)

- `halloween_bot/sim/kincal.py` — joint fit of link-length deltas, joint offsets, base tilt and the
  overhead camera. It fits against confirmed contacts, free descent paths, anchor poses, and photo
  silhouettes (Powell), with a holdout.
- `Calibration(zero={"shoulder_lift": "homing", ...})` — the per-joint zero convention, computed per
  arm from its own calibration file. Set in `geometry.json` `"zero"`.
- `apply_geometry()` — link lengths, claw length, base height/tilt, overhead camera pose.
