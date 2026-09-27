# Sim ↔ real kinematic calibration: status (2026-09-26)

## What's active in the sim

- **Joint mapping:** unchanged. Every joint is zeroed at the middle of its calibrated range (lerobot DEGREES
  convention), with no offsets.
- **Overhead camera:** re-fit (`halloween_bot/sim/geometry.json`). The C922 was bumped during the rig
  visit. Fitting the camera alone to the lit overlay photos raises the blue-arm silhouette IoU from
  0.345 to 0.643 (Δpos ≈ 3–4 cm, pitch +7°).

## Update: P0 frames settle the zero convention (2026-09-26, late)

The 1080 commanded measurement pose **P0** (all body joints 0, gripper 5) on both arms. The arms reached
lift 3.9 / 0.4 and elbow 3.7 / 3.7 (left / right). Frames and state are in `~/sim_ref/p0/`. No ruler
heights were taken.

- **The current mapping (A) is right; the "homing" zero (B) is rejected.** Both real wrist cameras look
  forward and slightly up, at a monitor across the room, with the claws pointing forward. A predicts
  that: forearm about level, claw tips 20–22 cm up. B predicts the claws pointing down at the table
  from 6–10 cm, which is clearly not what the cameras see. Side by side: `docs/sim/p0_compare.jpg`.
- **At full extension the right arm matches the sim closely.** The sim's blue-arm outline, rendered at
  the reached state with the re-fit camera, lies on the real arm within a few pixels: upper arm,
  forearm, wrist and claws (`docs/sim/p0_overlay.jpg`). So link lengths and lift/elbow offsets are
  not off by centimetres at this pose.
- **The one visible mismatch is the wrist-camera plate.** The real board is a flat plate sticking out
  on the arm's outer side (image-left), with the lens under it. Menagerie's mount sticks out
  diagonally on the inner side.

What this means for the table contacts: the fit that explained them (B) is ruled out, and the lift/elbow
chain matches at full extension. So the 3–11 cm gap at the contacts is not a zero-convention or
link-length error. Remaining candidates:
- what actually touches (the "what touches" question below; the real plate/lens differs from the model);
- a wrist_flex offset that only shows when the wrist is flexed (P0 has wrist_flex ≈ 0);
- compliance under table load.

A P1 ruler reading (elbow 20, the descent direction) and the plate/lens pose (card section C) would
separate them. Until then keep the CLAUDE.md warning: don't copy low-approach heights from the sim.

Two of those were then checked against the 9 confirmed contacts (right arm, current mapping):
- **A rigid part on the gripper doesn't fit.** At the contacts the claw axis is 40–62° from vertical, and
  "down" points mostly along the claws, not sideways. The single gripper-frame point that best
  touches at all 9 contacts sits 19 cm from the gripper origin, near no real part. It still misses
  by ±3 cm, sinks the rest pose 4.5 cm into the table, and cuts through free descent paths by 5 cm.
  A side plate would add at most ~2 cm of downward reach, not 3–11 cm.
- **A joint-scale error doesn't fit.** The scale comes from the encoder range (360°/4096 per tick,
  ≈1°/unit). Reaching 11 cm at a 25 cm lever needs ~25° at n ≈ −55, i.e. a ~45% scale error, and
  the rest pose (n = −86 / 95) would then be off by far more than the observed ≈1 cm.

So every direct view of the arm agrees with the current model: rest, the lit overlay photos, the
wrist-camera directions and P0. The contact rows are the outlier. The most decisive rig check is to
go back to the contact pose where the sim shows the most clearance (v3 row 7, 11 cm), stopping a few cm above
the recorded contact, and read the claw-tip height with a ruler. If it reads ~13 cm, the contacts
were not claw-on-tabletop at z = 0 (something on the table, or the detector firing on a load change).
If it reads ~0–2 cm, the model is wrong at descended poses.

**The contact photos already point to the first answer** (`docs/sim/contact_overlay.jpg`):
- **Nothing on the table.** The lit frames of all 9 contacts show nothing under the claws.
- **The claws are where the sim puts them.** At the three widest-gap rows (7, 3, 9), the current
  model's outline at the photo state lies on the real claws. That puts the tips 13, 12 and 5 cm up.
  The homing-zero outline sits visibly lower and doesn't match.
- **The camera is independently validated.** It was checked at P0, a pose outside the fit, so this
  isn't the camera absorbing a model error.

The 1080 confirms its detector can't tell a claw touch from the elbow losing to gravity. Every
contact fired on elbow tracking error while the descent drove the elbow into a growing gravity
moment. So the contact set is best read as "the elbow saturated here", not "the claws reached the
table". **Working conclusion: the current model is right at reach too, and the "3–11 cm high"
warning comes from false contacts.** The CLAUDE.md warning stays until the rig check (ruler or a
camera frame at row 7, stopped short) confirms it: relaxing a safety note on photo evidence alone is
premature.

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
