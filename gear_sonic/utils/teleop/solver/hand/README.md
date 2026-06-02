# G1 gripper hand poses (SONIC teleop)

How the Pico VR trigger maps to a Dex3-1 hand pose during SONIC teleop, and
how to tune it.

## Pipeline

```
Pico trigger ─► generate_finger_data()      (pico_manager_thread_server.py)
             ─► G1GripperInverseKinematicsSolver  (g1_gripper_ik_solver.py)
             ─► 7-DOF joint target ─► DDS rt/dex3/{left,right}/cmd ─► Dex3 hand
```

`generate_finger_data()` places one fingertip coincident with the thumb tip
when the trigger is pulled. The solver then picks the gesture whose finger is
closest to the thumb and returns that gesture's preset closed pose (interpolated
from open by the grip amount). Which fingertip `generate_finger_data()` sets is
therefore what selects the gesture.

## Dex3-1 joint order (firmware `motor_cmd[0..6]`)

The 7-DOF target maps straight through to the motors (no reordering in the C++
`setAllJointsCommand`):

| idx | joint    | motion                                            |
|-----|----------|---------------------------------------------------|
| 0   | thumb_0  | thumb **opposition** (swings sideways across palm)|
| 1   | thumb_1  | thumb **flex/curl**                               |
| 2   | thumb_2  | thumb **flex/curl**                               |
| 3   | index_0  | "motor-3" finger curl, base knuckle               |
| 4   | index_1  | "motor-3" finger curl, tip                        |
| 5   | middle_0 | "motor-5" finger curl, base knuckle               |
| 6   | middle_1 | "motor-5" finger curl, tip                        |

`q = 0` is fully open. Left/right hands mirror: the left-hand vector is built
with the signs below, and the right hand negates the whole vector
(`return q if side == "L" else -q`). Each joint has a hard limit; values past it
are clipped on the C++ side.

## Tuning knobs (`_get_index_close_q_desired`)

| variable        | joint(s) | controls                                                    |
|-----------------|----------|-------------------------------------------------------------|
| `amp0`          | q[0]     | thumb opposition — how far the thumb swings across (limit ~1.05) |
| `amp`           | q[1],q[2]| thumb flex — how far the thumb curls down toward the finger  |
| `ampA1`,`ampB1` | q[3],q[4]| motor-3 finger curl (knuckle, tip)                          |
| `ampA2`,`ampB2` | q[5],q[6]| motor-5 finger curl (knuckle, tip)                          |

Rules of thumb when the pinch doesn't close:
- thumb and finger not meeting vertically → raise `amp` (thumb curl).
- finger not folding in far enough → raise `ampA2`/`ampB2`.
- thumb and finger meet but laterally offset → adjust `amp0` (opposition).

## Current config — pinch (trigger)

The trigger now selects the **index gesture**: thumb opposes + curls, the
motor-5 finger curls to meet it, the motor-3 finger stays open → a clean
two-finger thumb-to-finger pinch.

- `generate_finger_data`: trigger sets the **index** fingertip (`4 + index`,
  index = 5) coincident with the thumb.
- `_get_index_close_q_desired`: `amp0 = 0.4`, `amp = 0.7`,
  `ampA1 = ampB1 = 0.0`, `ampA2 = ampB2 = 1.2`.
- Left-hand target: `[-0.4, 0.7, 0.7, 0.0, 0.0, -1.2, -1.2]` (right = mirror).

## Previous config — grasp

Previously the trigger selected the **middle gesture**: both fingers curl, no
thumb opposition → a raking three-finger grasp rather than a pinch.

- `generate_finger_data`: trigger set the **middle** fingertip (`4 + middle`,
  middle = 10).
- `_get_middle_close_q_desired`: `amp0 = 0.0`, `amp = 0.7`,
  `ampA1 = 1.0`, `ampB1 = 1.5`, `ampA2 = 1.0`, `ampB2 = 1.5`.
- Left-hand target: `[0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5]` (right = mirror).

`_get_middle_close_q_desired` is left unchanged — it is still used by
`run_vla_inference.py` for its closed-hand pose.
