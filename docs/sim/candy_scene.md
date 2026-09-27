# Candy scene (MuJoCo twin)

`SimEngine(scene="candy")`, `ctl start --sim --scene candy` or `SIM_SCENE=candy ./run_sim.sh`. The default
scene is still the toys-and-basket one that the π₀-FAST checkpoint was trained on.

## What's in it

- **Bowl:** wide and shallow, purple, 3.7 cm deep, 9 cm floor radius. It sits between the arms at
  (0.20, 0), jittered ±2 cm per reset.
- **Candy:** 11 types, 3–6 of them in the bowl per reset, with random positions, yaws and wrapper
  colours. The rest are parked on a hidden shelf under the table.

  | candy | notes |
  |---|---|
  | chocolate bar (10 cm), candy bar, licorice rope (11 cm), lollipop | long ones: the only candies eligible for a handover |
  | wrapped candy | |
  | candy corn | |
  | peanut butter cup, gumball, pumpkin candy | round |
  | candy box | |
  | gummy bear | |

  Colours are solid and matte, per the 1080: foil blows out the real wrist camera. Flat candies are
  fun-size thick (13–15 mm) so the SO101 jaws can grip them off a flat floor.
- **Plate:** the place target. It's placed per episode on the placing arm's outer side.
- **Person's hand:** a palm-up, slightly cupped child's hand with a forearm.
  - It's a dynamic, gravity-compensated body welded to a mocap target. A teleported mocap body has no
    velocity, so it couldn't carry a candy away in its palm.
  - It reaches in from the front once the robot has a candy. It takes it and withdraws.

## Tasks and instruction strings

The instruction names the candy (`{words}` = "candy corn", "peanut butter cup", ...).

| task | instruction | yield (scripted) |
|---|---|---|
| `pick_place` | `Pick up the {words} and put it on the plate.` | ~65% |
| `give_human` | `Give the {words} to the person.` | ~45% |
| `handover` | `Pick up the {words}, hand it to the other arm, and put it on the plate.` | ~0–3%, **not usable yet** |

Recording:

```bash
.venv/bin/python -m halloween_bot.sim.record --scene candy --tasks pick_place,give_human \
    --episodes 40 --root ~/lerobot/outputs/datasets/sim_candy_v0 --repo-id local/sim_candy_v0
```

- **Cameras and format:** same keys and units as the rig: `overhead`, `left_wrist`, `right_wrist`,
  12-d state and action, 30 fps. The π₀-FAST renames (`base_0_rgb`, ...) happen at policy time,
  as for real data.
- **Screening:** the sim is deterministic per seed, including with rendering. So every seed is first
  run headless (~0.4 s), and only successful ones are re-run with cameras. Low yields cost almost
  nothing.
- **Stats:** `meta/sim_record.json` has per-task yields and the seed, candy and arm of every episode.

## How the demonstrator grasps (`candy_expert.py`)

These are the things that mattered, in the order they were found:

1. **Aim the fixed jaw, not the midpoint.** Aim the fixed jaw's tip (its innermost point) 6 mm beside
   the candy, and open just wide enough for the moving jaw to clear the far side by 16 mm. Aiming the
   open-jaw midpoint makes the moving jaw shove small candy across the bowl.
2. **Servo onto the target.** The fitted servos sag several mm at the claw. `reach()` re-aims by the
   measured grasp-point error, like a teleoperator watching the wrist camera.
3. **Plan the descent by clearance.** The moving jaw swings like a pendulum and dips ~3.5 mm lowest
   mid-close. On the floor, friction pins it open against the weak gripper servo. So the descent
   stops where every jaw part clears the floor by 2.5 mm over the whole close.
4. **Wrist roll.** The jaws close across a long candy's short side, on the side with the most room
   from neighbouring candy and the bowl wall. Only candies with ≥4 mm of room for both jaws are
   chosen, as a teleoperator would.
5. **Squeeze.** Close 20 mm past the candy's width, and grip round candy at or above its equator.
   The fitted gripper is soft: ~1 N per 9 mm.

## Known limits

- **Gripper strength.** The dynamics fit only saw free motion (kp 0.49, 0.47 Nm limit), so its grip
  strength is unidentified and probably weaker than the real STS3215. A stiffer gripper
  (kp 3–8, 1.5–2 Nm) didn't change pick-and-place yield, so the fitted values stay.
- **Handover doesn't work yet.** What was tried:
  - claws-forward presentation: drops the candy, because the pivoting jaw squeezes it out of a tip
    grip as the wrist pitches;
  - claws-down side by side: the grippers and camera mounts collide;
  - claws-forward end pinch: the pivoting jaw grips only the candy's edge.

  A usable handover probably needs a deeper grip than a bowl floor allows (a regrasp on a stand),
  or a real teleop demo to copy.
