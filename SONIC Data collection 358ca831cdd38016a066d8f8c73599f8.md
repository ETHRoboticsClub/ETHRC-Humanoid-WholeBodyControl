# SONIC Data collection

<aside>
⚡

Plan of action for recording **LeRobot v2.1 datasets** with the **SONIC unified WBC policy** on the **Unitree G1**, using **PICO whole-body teleop in POSE mode** with the **ETH Robotics Club RealSense ego camera**. Sibling of the WBC Stack page (decoupled WBC); produces datasets that are drop-in compatible with **Isaac-GR00T** post-training of GR00T N1.6.

</aside>

# Goal

Record LeRobot v2.1 datasets in which the **entire G1 body** is teleoperated by full-body SMPL streaming from the PICO via SONIC's POSE mode. The dataset format matches what Isaac-GR00T expects — so the fine-tune deltas vs. the decoupled-WBC version come from the *quality* of the action labels (coordinated whole-body motion, not upper-body teleop overlaid on an RL lower body).

# Plan of Action — Phases

<aside>
🗺️

Five phases. Phases 0–2 are one-time setup. Phases 4–5 are the loop you repeat every recording session. Click any phase below to expand its details.

</aside>

| Phase | Purpose | Frequency |
| --- | --- | --- |
| 0 | Prerequisites & sanity checks | Once |
| 1 | Workstation one-time install | Once per workstation |
| 2 | Orin RealSense camera server install | Once per robot |
| 3 | Sim2Sim dry run *(optional)* | Once before first real session |
| 4 | Real-robot recording session | Every session |
| 5 | Post-processing & merging | After each batch |

---

# Phase 0 — Prerequisites

- ✅ `GR00T-WholeBodyControl` repo cloned on the workstation; deployment Quick Start completed (sim2sim runs end-to-end).
- ✅ VR Teleop setup completed: PICO calibrated, `.venv_teleop` ready, XRoboToolkit installed.
- ✅ SONIC checkpoints downloaded (HuggingFace `nvidia/GEAR-SONIC` + sample motion data).
- ✅ G1 hardware in working order (same setup as the decoupled stack).
- ✅ Workstation: Ubuntu 24.04, recent NVIDIA driver, `uv` available.
- ✅ Tight-fitting pants/leggings on hand for the operator (mandatory for foot trackers).
- ✅ The decoupled-WBC RealSense camera server already runs successfully on the Orin (confirms USB/udev/firmware are healthy — we will *not* reuse the server itself, just the working hardware path).

# Phase 1 — Workstation one-time install

<aside>
📦

Creates `.venv_data_collection`, a third venv alongside `.venv_teleop` and `.venv_sim`. Heavy ML deps (LeRobot, PyAV, OpenCV) live here so they don't pollute the teleop env.

</aside>

From the repo root on the workstation:

```bash
bash install_scripts/install_data_collection.sh
```

Installs `gear_sonic[data_collection]` plus `espeak` (system) for voice feedback. Quick verify:

```bash
source .venv_data_collection/bin/activate
python -c "import lerobot, av, cv2; print('ok')"
```

# Phase 2 — Orin RealSense camera server install

<aside>
📷

The **only** piece that runs on the robot. The bundled `composed_camera` server already supports the Intel RealSense — no driver patching needed. Steps: install the camera venv, add `pyrealsense2`, register the systemd service, verify the feed from the workstation.

</aside>

## Step 2.1 — Run the camera install script on the Orin

```bash
ssh unitree@192.168.123.164
cd ~/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

The script creates `.venv_camera` (uv-managed Python 3.10) and installs `gear_sonic[camera]` (DepthAI, ZMQ, msgpack, OpenCV, tyro). It does **not** install `pyrealsense2` — that's the next step. On first run, decline the systemd prompt (`N`) so we can install `pyrealsense2` first; we'll come back and enable systemd in Step 2.3.

## Step 2.2 — Install `pyrealsense2` and confirm the camera is visible

```bash
source .venv_camera/bin/activate
uv pip install pyrealsense2
```

<aside>
⚠️

The venv is created with `uv venv`, which omits a real `pip` by design — use `uv pip install`. If miniconda's `(base)` environment is active, `which pip` may resolve to conda's pip (Python 3.13) and silently install into the wrong interpreter; `pyrealsense2` has no aarch64 wheel for 3.13. Either `conda deactivate` before installing, or always go through `uv pip install` / `.venv_camera/bin/python -m pip install`.

</aside>

Verify the camera enumerates and capture its serial number:

```bash
python -c "import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().devices])"
```

Expected output: `[('Intel RealSense D435I', '<SERIAL>')]`. Record the serial — the install script asks for it in the next step.

If the list is empty: confirm the USB cable is on a USB 3.x port, check `dmesg | grep -i realsense`, and (one-time, only if needed) install udev rules: `sudo cp /opt/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger`.

## Step 2.3 — Register the systemd service

Re-run the install script, this time accepting the systemd prompt:

```bash
bash install_scripts/install_camera_server.sh
```

Interactive answers:

- OAK detection retry: `n` (we don't have OAK).
- Ego-view camera type: `realsense`.
- Ego-view device ID: `<SERIAL>` from Step 2.2.
- Add left-wrist / right-wrist camera: `N` / `N`.
- ZMQ port: press Enter for the default `5555`.
- ExecStart preview: confirm with `Y`.
- Install as systemd service: `Y` (will prompt for sudo).

Confirm the service is up:

```bash
sudo systemctl status composed_camera_server.service
journalctl -u composed_camera_server.service -f
```

Expected: `Active: active (running)`. The journal shows the RealSense initializing and frames being published.

<aside>
⚠️

If the decoupled `start_camera_server.sh` (or its systemd wrapper) is also active on this Orin, only one process can bind port 5555. Either disable the decoupled side (`sudo systemctl disable --now <decoupled-service-name>`) or change one of the ports. Confirm with `pgrep -af "videohub_pc4|realsense.*--server|composed_camera"`.

</aside>

## Step 2.4 — Verify the feed from the workstation with `run_camera_viewer.py`

While the Orin service is running, on the workstation:

```bash
cd ~/ETHRC-Humanoid-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 --camera-port 5555
```

A tiled OpenCV window opens. Pass criteria:

- Live ego-view RGB feed at 640×480, ~30 Hz.
- Colors look correct (no R/B swap).
- Latency log lines (`Image latency for ego_view: ~XX ms`) appear and stay reasonable.

You will also see a depth stream rendered alongside the color view — that's the RealSense publishing depth on top of color. The data exporter consumes only `ego_view` (color) and ignores everything else, so this is harmless.

Reboot test: `sudo reboot` the Orin, then re-run the viewer command — the feed should come up automatically without any manual intervention on the Orin.

## Day-to-day commands for the camera service

- **SONIC camera-server systemd commands** (click to expand)
    
    The camera server runs as a systemd unit (`composed_camera_server.service`) and starts automatically on every Orin boot — you don't need to run anything by hand to bring it up.
    
    - Check status: `sudo systemctl status composed_camera_server.service`
    - Follow logs live: `journalctl -u composed_camera_server.service -f`
    - Last 200 lines from the journal: `journalctl -u composed_camera_server.service --no-pager -n 200`
    - Restart (e.g. after USB unplug, or to recover from a stuck state): `sudo systemctl restart composed_camera_server.service`
    - Stop temporarily (e.g. to use the decoupled `start_camera_server.sh` instead): `sudo systemctl stop composed_camera_server.service`
    - Start again after a stop: `sudo systemctl start composed_camera_server.service`
    - Disable autostart on boot: `sudo systemctl disable composed_camera_server.service`
    - Re-enable autostart: `sudo systemctl enable --now composed_camera_server.service`
    - See who's holding port 5555 / the camera: `pgrep -af "videohub_pc4|realsense.*--server|composed_camera"`
    
    ⚠️ Only one camera server can bind port 5555 at a time. Stop the SONIC service before launching the decoupled `start_camera_server.sh`, and vice versa.
    

# Phase 3 — Sim2Sim verification *(optional)*

Optional dry-run against MuJoCo before touching real hardware. Skip if you'd rather validate directly on the G1.

```bash
python gear_sonic/scripts/launch_data_collection.py --sim
```

Spawns a sim window plus a 4-pane tmux session (C++ deploy / PICO teleop / data exporter / camera viewer). Confirm:

- The MuJoCo G1 enters POSE mode and follows your body smoothly.
- **Left Grip + A** toggles recording (the data exporter pane logs episode start/stop, espeak announces).
- An episode appears under `outputs/<dataset-name>/` with parquet + mp4.

Note: in `--sim` the camera comes from MuJoCo's offscreen rendering, not from the Orin's RealSense — this validates the SMPL → policy → exporter chain only. RealSense is validated end-to-end in Phase 4.

---

# Phase 4 — Real-robot recording session

<aside>
🤖

The loop you'll run **every session**. Designed to be muscle-memorizable — six steps from boot to recording.

</aside>

## Step 4.1 — Hardware bring-up

1. **Power on the G1, enter Developer Mode** with the physical controller: `L2+R2`, then `L2+A`, then `L2+B`. Robot announces *"Developer mode"*.
2. **Connect ethernet, configure the workstation IP, verify connectivity**:

```bash
sudo ip addr add 192.168.123.222/24 dev enp6s0
sudo ip link set enp6s0 up
ping 192.168.123.164
```

1. **Confirm the Orin RealSense camera server is up** (systemd should have started it on boot):

```bash
ssh unitree@192.168.123.164 "sudo systemctl status composed_camera_server.service"
```

1. **PICO + workstation networking**:
    - Both on the same LAN (phone hotspot or shared WiFi).
    - XRoboToolkit on PICO shows **"Working"** next to the workstation IP — re-press **Connect** if not.
    - **"Controller"** info enabled in the app UI.
2. **Start the workstation companion app**: launch XRoboToolkit-PC-Service.

## Step 4.2 — Launch the data collection stack

<aside>
▶️

One command. Native (no Docker). Equivalent role to `deploy_g1.py` in the decoupled stack.

</aside>

From the repo root on the workstation:

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "your task description here" \
    --data-exporter-frequency 50
```

Useful additional flags:

| Flag | Use when… |
| --- | --- |
| `--dataset-name <name>` | You want to **append** episodes to an existing dataset instead of starting a new one. |
| `--no-text-to-speech` | espeak announcements are distracting. |
| `--no-camera-viewer` | You don't need pane 3 for monitoring. |
| `--record-wrist-cameras` | Only if you've added wrist OAKs / wrist RealSenses (skip for the single-ego setup). |

The launcher attaches you to a **4-pane tmux session**:

```jsx
┌──────────────────────────┬──────────────────────────┐
│ Pane 0: C++ deployment   │ Pane 2: data exporter    │
│ (gear_sonic_deploy)      │ (.venv_data_collection)  │
├──────────────────────────┼──────────────────────────┤
│ Pane 1: PICO teleop      │ Pane 3: camera viewer    │
│ (.venv_teleop)           │ (.venv_data_collection)  │
└──────────────────────────┴──────────────────────────┘
```

Wait for pane 0 to print **`Init done`** before doing anything on the PICO. Switch panes with `Ctrl+b` then arrow keys.

## Step 4.3 — Data exporter setup

In pane 2, you'll be prompted for:

1. **Dataset name** — accept the auto-generated timestamp, or pass an existing name to append.
2. **Append vs. new dataset** — confirm the choice.

This happens once at session start.

## Step 4.4 — Engage SONIC and enter POSE mode

<aside>
⚠️

Before pressing anything: stand in the **calibration pose**. Feet together, upper arms hanging straight down, forearms bent **90° forward** (L-shape at each elbow), palms facing inward. Tight-fitting pants — non-negotiable for foot tracker line-of-sight.

</aside>

PICO sequence:

| # | Action | What happens |
| --- | --- | --- |
| 1 | Stand in calibration pose | Body matches the all-zero reference frame |
| 2 | **A + B + X + Y** simultaneously | Engages policy + runs **CALIB_FULL** (head + both wrists). Robot enters PLANNER mode. |
| 3 | Align body with robot's current pose | Avoids snap-jerk on next step |
| 4 | **A + X** | Enters **POSE** mode. Whole body follows your SMPL stream. |

At the end of step 4 the robot is mirroring you in real time, ready to record.

## Step 4.5 — Record episodes

| Action | Buttons |
| --- | --- |
| Start episode | **Left Grip + A** |
| Stop & save episode | **Left Grip + A** (again) |
| Discard current episode | **Left Grip + B** |
| Toggle from keyboard (alt.) | `c` (toggle) / `x` (discard) — sent over ZMQ port 5580 |

<aside>
⚠️

Different from decoupled WBC! There you used a bare `A` to start/stop an episode. Here `A` alone is part of the engagement combos, so episode control needs the **Left Grip** modifier.

</aside>

Per-episode loop:

1. Reset the scene; position the robot via your body (still in POSE).
2. **Left Grip + A** → recording starts (espeak announces).
3. Execute the task end-to-end.
4. **Left Grip + A** → recording stops, episode saved.
5. If the take was bad: **Left Grip + B** *before* stopping, to discard.
6. Repeat.

## Step 4.6 — End the session

| Action | Buttons / Command |
| --- | --- |
| Exit POSE → PLANNER (idle) | **A + X** |
| Stop policy | **A + B + X + Y** → OFF |
| **Emergency stop** (any time) | **A + B + X + Y** on PICO, or **`O`** in the C++ pane |
| Detach tmux (keep running) | `Ctrl+b`, then `d` |
| Reattach later | `tmux attach -t sonic_data_collection` |
| Kill tmux session | `Ctrl+\` in any pane |

---

# Phase 5 — Post-processing & merging

<aside>
🧹

Always run this **before** fine-tuning. It removes stale-SMPL frames (zero-pose during operator pauses or ZMQ drops) plus the consecutive frozen lead-in frames preceding them. Without it, the action labels include stale targets that hurt the fine-tune.

</aside>

Clean a single dataset (non-destructive):

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned
```

Merge multiple sessions (script_config consistency is validated):

```bash
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/session1 outputs/session2 outputs/session3 \
    --output-path outputs/merged_dataset
```

Or via list file (one path per line, `#` for comments):

```bash
python gear_sonic/scripts/process_dataset.py \
    --dataset-list datasets.txt \
    --output-path outputs/merged_dataset
```

SMPL cleaning is applied by default during merging. Skip with `--no-remove-stale-smpl` only if you have a reason.

---

# POSE Mode Reference

## Recorded data channels

| Feature | Shape | Meaning |
| --- | --- | --- |
| `observation.state.joint_position` | (N,) | Actuated joint positions (rad) |
| `observation.state.joint_velocity` | (N,) | Actuated joint velocities (rad/s) |
| `observation.state.body_rotation_6d` | (6,) | Base orientation (6D rotation) |
| `observation.state.projected_gravity` | (3,) | Gravity vector in body frame |
| `observation.images.ego_view` | (480, 640, 3) | Ego camera image (mp4-encoded video) — RealSense color in our setup |
| `action.joint_position` | (N,) | Teleop target joint positions |
| `action.body_rotation_6d` | (6,) | Teleop target body rotation |
| `annotation.human.action.task_description` | str | `--task-prompt` for this episode |

**Recording rate:** the LeRobot dataset is recorded at **50 Hz** (state, actions, video) regardless of camera fps. The RealSense publishes at 30 fps; the data exporter ticks at 50 Hz and reuses the latest frame when no new one has arrived (~1 duplicate every 1.7 ticks). This is intentional — don't try to push the camera higher unless you actually see motion blur or stutter.

## Output layout

```jsx
outputs/<dataset-name>/
├── data/
│   └── train-00000.parquet      # joint states, actions, annotations
├── videos/
│   └── observation.images.ego_view/
│       └── episode_000000.mp4   # H264-encoded ego camera
└── meta/
    ├── info.json                # fps, features, sizes
    ├── modality.json            # GR00T modality config
    ├── episodes.jsonl           # per-episode metadata
    └── tasks.jsonl              # task prompts
```

## Fine-tuning hand-off to Isaac-GR00T

```bash
# In the Isaac-GR00T repo
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/outputs/<dataset-name>
```

---

# Mode-switching safety

<aside>
⚠️

Before pressing **A + X** to enter POSE: **align your body with the robot's current pose**. POSE mode snaps the robot to your physical stance — a large mismatch produces aggressive, dangerous motion.

</aside>

Recovery if calibration goes wrong:

1. **A + X** → exit POSE.
2. Re-stand in the calibration pose.
3. **A + B + X + Y** twice (off then on) to redo `CALIB_FULL`.
4. Re-align with the robot, then **A + X** back into POSE.

---

# Gotchas

- ⚠️ **Different recording bindings vs decoupled WBC.** It's **Left Grip + A** here, not bare `A`. Easy to slip up if you run a decoupled session and a SONIC session in the same day.
- ⚠️ **Tight clothing required.** Loose pants will lose foot tracker line-of-sight; the SMPL stream goes erratic and the safety story breaks.
- ⚠️ **No Docker.** Native venvs only. Don't try to launch SONIC from inside the decoupled Docker container.
- ⚠️ **Camera-server coexistence.** If you keep the decoupled `start_camera_server.sh` around, only one process can bind port 5555. Easiest: `sudo systemctl disable --now <decoupled-service-name>` (or change one of the ports).
- ⚠️ **`pip` shadowing inside conda.** With miniconda's `(base)` active, `which pip` can resolve to conda's pip (Python 3.13) instead of the activated venv. Always use `uv pip install` or `.venv_camera/bin/python -m pip install` when installing into the camera venv.
- ⚠️ **Depth is published but ignored.** The bundled SONIC RealSense driver enables both color and depth and publishes both; the data exporter only consumes `ego_view` (color). The viewer renders depth alongside color — that's normal.
- ⚠️ **PICO XRoboToolkit IP** must point at the workstation, not a stale address from a previous session.
- ⚠️ **The data exporter assumes POSE mode for VLA-grade actions.** Recording in PLANNER would produce action labels with frozen-or-planner-driven upper bodies, which is *not* what you want for N1.6 fine-tuning.

---

# Quick Reference Card

| Command | Purpose |
| --- | --- |
| `bash install_scripts/install_data_collection.sh` | One-time workstation install (Phase 1) |
| `bash install_scripts/install_camera_server.sh` | One-time Orin install — base venv + systemd (Phase 2.1, 2.3) |
| `uv pip install pyrealsense2` (in `.venv_camera`) | RealSense Python wrapper (Phase 2.2) |
| `sudo systemctl status composed_camera_server.service` | Check camera autostart (Phase 2.3) |
| `python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555` | Verify the camera feed from the workstation (Phase 2.4) |
| `python gear_sonic/scripts/launch_data_collection.py --sim` | Sim2Sim dry run (Phase 3) |
| `python gear_sonic/scripts/launch_data_collection.py --camera-host 192.168.123.164 --task-prompt "..."` | Real-robot session (Phase 4) |
| `python gear_sonic/scripts/process_dataset.py --dataset-path ... --output-path ...` | Clean / merge datasets (Phase 5) |
| `tmux attach -t sonic_data_collection` | Re-attach after detaching |

---

# PICO Cheatsheet

| Goal | Buttons |
| --- | --- |
| Engage policy + CALIB_FULL | **A + B + X + Y** |
| Enter POSE mode | **A + X** (from PLANNER) |
| Exit POSE → PLANNER | **A + X** (again) |
| Start / stop episode | **Left Grip + A** |
| Discard current episode | **Left Grip + B** |
| Hand grasp | **Trigger** (per hand) |
| Emergency stop | **A + B + X + Y** → OFF |

---

# 🔧 Runaway-motion investigation (2026-05-06)

<aside>
⚠️

**Status:** root cause strongly suspected, not yet patched. Operational workaround proposed. **Do not engage the policy without a partial safety harness** until Followup item 1 below is verified or upstream patch lands.

</aside>

## Symptom

Immediately after pressing **A + B + X + Y** to engage the policy from PLANNER mode, the G1 makes fast, seemingly-random, large-amplitude joint movements. Behavior is dangerous; we Ctrl+Z'd the C++ deploy each time. No PICO POSE-mode (A+X) was needed to trigger — the violent motion happens on the engagement combo alone.

## TL;DR root cause

At startup, the SONIC C++ deploy sets `current_motion_` to the **first reference motion in the load order**, which on this machine is `macarena_001__A545` — a dance recording whose frame 0 is *not* a neutral pose (the heading-reset quaternion in the log, `(-0.670, -0.001, -0.013, 0.742)`, is an ≈84° body rotation). When **A + B + X + Y** transitions the deploy into CONTROL state, the planner takes ≈11 ms to initialize. The control loop runs at 50 Hz (20 ms tick). At least the first post-engagement tick consumes macarena frame 0 as the motion-encoder conditioning target, so the policy commands the robot to snap into a dance pose. Combined with high `kp`, no soft-start ramp, no torque limiter, and `Compliance control: DISABLED`, this single dance-pose target produces a torque impulse that kicks the controlled system into a large-amplitude oscillation that can take many seconds to settle (or never settles in practice if joint limits / self-collision feed back into more error).

**This is a one-frame torque impulse, not a one-frame OOD perception** — the policy itself stays in-distribution; it's the closed-loop system that gets kicked hard at t=0.

## Evidence trail

- **Code references that nail down the race**
    - **Default `current_motion_` is index 0** of the load order: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:2267-2271`
        
        ```cpp
        motion_reader_.current_motion_index_ = 0;
        current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
        ```
        
    - **Swap to planner_motion happens later, at line 3290:** `current_motion_ = planner_motion_;` — inside the planner state-update path. Until that line executes, `current_motion_` is whatever was loaded first.
    - **Policy is conditioned on `current_motion_`** via `policy/release/observation_config.yaml`: `motion_joint_positions_10frame_step5`, `motion_anchor_orientation`, `motion_joint_positions_lowerbody_10frame_step5`, etc. — all sourced from `current_motion_`.
    - **Engagement-time log shows two heading resets with very different anchor quaternions** (~0.7 component delta), one for macarena, one for planner_motion. That is the visible fingerprint of the swap, and confirms macarena was active for at least one tick of state evaluation.
    - **Planner init timing:** `Planner Init timing - Model: 11327us, Extract: 24us` (from your log). 11.3 ms < 20 ms control tick — a coin flip whether one tick lands on macarena.
    - **No safety ramp** in the deploy: agent search found no `soft_start`, `torque_ramp`, `easing`, `deadman`, or `estop` gating in `gear_sonic_deploy/src/`. PD gains are applied at full magnitude on the first tick.
    - **Compliance disabled** is the released non-compliant checkpoint's normal state (`g1_deploy_onnx_ref.cpp:2529-2539`), but it removes the impedance damping that would otherwise mask a small impulse.
- **Why OOD-from-1-frame is *not* the mechanism**
    
    The observation vector is dominated by current state (joint positions, velocities, IMU, gravity) plus motion-encoder targets. One tick of macarena conditioning shifts joint state by 1–2° and adds a single spike to `his_last_actions_10frame_step1`. The training distribution easily covers this. Subsequent ticks see a slightly perturbed state but valid observations — the policy keeps emitting reasonable actions aimed at the (now correct) idle target. The damage is done by the **closed-loop dynamics** kicked by the first tick's torque impulse, not by the policy emitting nonsense thereafter.
    

## What was ruled out

- **Candidates investigated and eliminated**
    - **Concurrent controller / dual command publisher.** Workstation `pgrep` clean; Orin `pgrep` clean (only `composed_camera_server` running, expected). No `run_g1_control_loop.py`, no `deploy_g1.py`, no Docker containers, no decoupled-WBC autostart unit, no stray crontab. Diagnostic script saved at `diagnose_orin.sh` in the repo root — run it again before any future debug session: `ssh unitree@192.168.123.164 'bash -s' < diagnose_orin.sh`.
    - **DDS / ROS env contamination.** No `ROS_DOMAIN_ID`, `CYCLONEDDS_URI`, `RMW_IMPLEMENTATION` set on either side. `setup_env.sh` sets `ROS_LOCALHOST_ONLY=1` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` cleanly.
    - **G1 type mismatch / variant ID 5.** `g1_deploy_onnx_ref.cpp:2639-2641, 2661` shows `mode_machine` is *echoed back* from low-state to LowCmd — it's not a validated variant identifier. Type 5 is just whatever the robot firmware reports. The 29-DoF body + 2 Dex3 hands setup matches SONIC's hardcoded `G1_NUM_MOTOR = 29` (`robot_parameters.hpp:32`), and hands are on a separate channel (`rt/dex3/{left,right}/cmd` per `dex3_hands.hpp:75-79`).
    - **Stale TensorRT engines.** `policy_model_decoder.trt` and `encoder_model_encoder.trt` were compiled at 15:14 from `model_*.onnx` placed at 15:07 — 7-minute gap, so consistent. No staleness.
    - **`--output-type all` activating multiple commanders.** Per code read of `g1_deploy_onnx_ref.cpp:2545-2574`, `all` only adds **telemetry** sinks (ZMQ debug + ROS2 telemetry). Motor commands path is independent and always active via `lowcmd_publisher_->Write()` at 500 Hz.
    - **DoF count mismatch.** 29-DoF body + 14-DoF (2×7) Dex3 hands is the canonical SONIC G1 setup. Hands published separately, not part of `rt/lowcmd`.

## Workarounds (until upstream patch)

1. **Use a *partial* safety harness, not a full rope-hang.** The released SONIC policy is fully proprioceptive (no foot-contact / GRF observation), so suspending doesn't directly trigger runaway, but it removes the floor's natural damping for any other instability. Keep the feet lightly weighted on the floor at engagement.
2. **Hand-position the robot close to its default standing pose before engaging.** Smaller error at t=0 → smaller initial torque impulse if the macarena tick fires.
3. **Stay in PLANNER mode for ≥10 seconds before pressing A+X.** If the macarena tick fired, you'll see the kick in PLANNER alone — disengage immediately (A+B+X+Y → OFF) before going to POSE mode.
4. **Verify the SMPL stream is alive *before* engaging.** Quick subscriber on the workstation:
    
    ```bash
    python -c "
    import zmq, msgpack
    ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
    s.connect('tcp://localhost:5556'); s.setsockopt_string(zmq.SUBSCRIBE, '')
    for i in range(20):
        if s.poll(500):
            topic, payload = s.recv_multipart()
            d = msgpack.unpackb(payload, raw=False)
            print(topic.decode(), '|', list(d.keys()) if isinstance(d, dict) else type(d))
        else:
            print('(no data — XRoboToolkit-PC-Service likely not publishing)'); break
    "
    ```
    
    You should see `pose`, `command`, and `planner` topics with sensible-looking dicts. If only `command` appears or the loop times out, the PICO side isn't streaming — don't engage.
    
5. **Always re-run `diagnose_orin.sh`** before a fresh session to confirm no stray controllers came back since last time.

## Followup work — upstream patches to discuss with the SONIC owners

Both touch the **don't-touch list**, so coordinate with whoever maintains `gear_sonic_deploy/` before opening a PR.

1. **Reorder the loaded reference motions so a static / neutral pose is index 0.** The cheapest fix. The `reference/example/` set we have today doesn't include a true "neutral standing" recording, but `squat_001__A359` may have a static intro frame — worth verifying its frame 0. Alternative: synthesise a one-frame "rest pose" CSV with `default_angles` from `policy_parameters.hpp` and place it first.
2. **Gate the CONTROL state transition on `current_motion_ == planner_motion_`.** A few-line patch around `g1_deploy_onnx_ref.cpp:3821` that refuses to enter CONTROL until the planner has swapped `current_motion_`. Eliminates the race entirely. This is the right structural fix and the **recommended approach** — it doesn't change any data, doesn't change policy semantics, just adds a guard that delays engagement by ≈11 ms (until the planner publishes its first frame). Concrete diff in the toggle below.
- **Proposed patch — Option A diff (recommended fix)**
    
    In `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, around line 3821 inside the `ProgramState::WAIT_FOR_CONTROL` branch. **Before:**
    
    ```cpp
    for (auto& oi : output_interfaces_) { if (oi) oi->publish_config(); }
    if (operator_state.start) {
      // Warn if starting control in token mode without tokens, but allow it
      if (initial_encoder_mode_ == -1 && !first_token_received_) {
        static int warn_count = 0;
        if (warn_count % 50 == 0) {
          std::cout << "⚠⚠⚠ [Token Safety] WARNING: Starting control before tokens arrive! "
                    << "Robot will use zero tokens until tokens start streaming." << std::endl;
        }
        warn_count++;
      }
      std::cout << "[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state" << std::endl;
      program_state_ = ProgramState::CONTROL;
    }
    ```
    
    **After:**
    
    ```cpp
    for (auto& oi : output_interfaces_) { if (oi) oi->publish_config(); }
    if (operator_state.start) {
      // Warn if starting control in token mode without tokens, but allow it
      if (initial_encoder_mode_ == -1 && !first_token_received_) {
        static int warn_count = 0;
        if (warn_count % 50 == 0) {
          std::cout << "⚠⚠⚠ [Token Safety] WARNING: Starting control before tokens arrive! "
                    << "Robot will use zero tokens until tokens start streaming." << std::endl;
        }
        warn_count++;
      }
      // Safety gate: refuse to enter CONTROL until the planner has taken over
      // current_motion_. Otherwise the first 1-2 control ticks would condition
      // the policy on macarena (or whatever motion is index 0 in the load order),
      // producing a torque impulse that destabilises the closed loop.
      {
        std::lock_guard<std::mutex> lock(current_motion_mutex_);
        if (current_motion_ != planner_motion_) {
          static int planner_wait_count = 0;
          if (planner_wait_count++ % 50 == 0) {
            std::cout << "[Control] Waiting for planner to take over current_motion_ "
                      << "before entering CONTROL..." << std::endl;
          }
          break;
        }
      }
      std::cout << "[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state" << std::endl;
      program_state_ = ProgramState::CONTROL;
    }
    ```
    
    **Behaviour after patch:** pressing **A + B + X + Y** still triggers the engagement, but the deploy busy-loops in `WAIT_FOR_CONTROL` (printing `[Control] Waiting for planner to take over current_motion_...` once per second) until the planner thread executes the swap at line 3290 (≈11 ms after engagement). Then it transitions to CONTROL on the next iteration, with the policy correctly conditioned on `planner_motion_` from the very first tick.
    
    **Why this fixes the runaway:** by construction, no control tick can ever be conditioned on macarena (or any other reference motion) at the moment of CONTROL entry. The torque impulse described in the TL;DR is impossible.
    
    **Caveats / verify before merging:**
    
    - Confirm `current_motion_mutex_` is the right lock for this read — grep usage to make sure no other path holds it for long.
    - Confirm `planner_motion_` is non-null at this point in the lifecycle (allocated at line 2240 before `WAIT_FOR_CONTROL` is reachable, so should be safe).
    - If the planner thread fails to ever produce a frame (e.g. crashed planner ONNX), this would block engagement forever. Consider adding a timeout (e.g. log a fatal error after 5 s and exit) so failures are loud rather than silent.
    - Test in `--sim` first: `python launch_data_collection.py --sim`, engage, confirm the wait message appears at most a few times before CONTROL transitions, and that the simulated robot does not lurch.
    
    **Suggested branch:** `fix/control-state-planner-gate`. Suggested commit title: `fix(deploy): gate CONTROL state on planner_motion swap to prevent macarena-frame-0 torque impulse at engagement`. Worth upstreaming as a PR to NVlabs/GR00T-WholeBodyControl — same race exists there.
    
    **Constraint reminder:** `gear_sonic_deploy/` is on the don't-touch list per the original integration plan. This patch is justified by the safety impact, but should be coordinated with the SONIC owners before merging.
    
1. **Add a soft-start torque ramp** — e.g. linearly scale `kp` and `kd` from 0 to nominal over the first 500 ms of CONTROL. Defends against this *and* future related issues (e.g. POSE-mode entry from a non-aligned operator pose). Touches `g1_deploy_onnx_ref.cpp:3115-3127` and the LowCommandWriter path.
2. **Add an SMPL-stream / calibration health check** before allowing CONTROL transition. Currently the C++ deploy ingests whatever the ZMQManager hands it without verifying calibration completed.

## Artifacts

- **Local diagnostic script:** `diagnose_orin.sh` in the repo root. Read-only sweep of the Orin's processes, services, DDS env, listeners, journal. Run it as `ssh unitree@192.168.123.164 'bash -s' < diagnose_orin.sh`.
- **Engagement log captured at the moment of the runaway** — the two consecutive `Reset init reference data root rotation` lines with ≈0.7 quaternion-component delta are the diagnostic fingerprint of this race. Future debugging: if you see those again with very different quats, the race fired.

## Open questions for whoever takes over

- Is the released `nvidia/GEAR-SONIC` decoder ONNX *intended* to support engagement-time conditioning on arbitrary reference motions, or only on planner_motion? The HF model card doesn't say. If it's planner-only, the workstation deploy should refuse to load other motions at startup (or at least not set them as `current_motion_`).
- Is there a recommended "warm-up" mode in the SONIC stack we missed — a way to run the policy in a torque-zero "observation only" pass for the first 1–2 ticks to populate the action history without commanding the robot? Worth asking on the SONIC repo issue tracker.
- Can we reproduce the kick in `--sim` (MuJoCo)? If so, the bug is purely software and we can iterate on patches without robot risk.

# Resources

- **SONIC paper**: [https://arxiv.org/abs/2511.07820](https://arxiv.org/abs/2511.07820)
- **GR00T-WholeBodyControl docs**: [https://nvlabs.github.io/GR00T-WholeBodyControl/](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- **Data Collection for VLA tutorial**: [https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)
- **PICO VR Whole-body Teleop tutorial**: [https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)
- **Whole-body Teleoperation best-practices**: [https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/teleoperation.html](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/teleoperation.html)
- **GitHub repo**: [https://github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
- **Sibling page**: WBC Stack (decoupled WBC version)