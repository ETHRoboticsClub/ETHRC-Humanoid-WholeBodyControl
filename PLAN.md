# SONIC Stack Integration — Implementation Plan

**Mission:** Integrate ETH Robotics Club's existing Intel RealSense ego camera into the NVIDIA GEAR-SONIC whole-body control stack on the Unitree G1, so we can record LeRobot v2.1 datasets via PICO whole-body teleop (POSE mode) and fine-tune GR00T N1.6.

**Source of truth:** Notion page "SONIC Stack" (`357a7dd2-d1f3-81dd-a469-d7ab733972e0`). Re-fetch via Notion MCP if any guidance here looks stale.

**Strategy:** A — fix bit-rot in the bundled SONIC RealSense driver. Fall back to C (custom thin publisher) only with explicit user agreement. Do NOT pursue B.

---

## Repo / environment notes

- **Workstation repo path (actual):** `/home/ethrc-thor/ETHRC-Humanoid-WholeBodyControl` — ETH Robotics Club fork of `NVlabs/GR00T-WholeBodyControl`. Notion page references `~/GR00T-WholeBodyControl` generically; internal layout is identical, so all relative paths in Notion apply unchanged. Treat this dir as "the workstation repo."
- **Orin repo path (per Notion):** `~/GR00T-WholeBodyControl` on `unitree@192.168.123.164`. To be confirmed on the Orin in P2.1.
- **Active git branch:** `feat/sonic-data-collection`. Driver work will branch from here as `feat/realsense-camera-sonic` (P2.0).
- **Don't-touch list (hard constraint):** `decoupled_wbc/`, `gear_sonic_deploy/`, `gear_sonic/scripts/run_data_exporter.py`, `gear_sonic/scripts/pico_manager_thread_server.py`, image shape `(480, 640, 3)`, depth/IR streams.

---

## Camera-server audit findings (2026-05-05) — read before working Phase 2

The bundled RealSense path is **wired through `composed_camera.py` but was never integration-tested**. It is *not* "bit-rot in `pyrealsense2` API"; the API calls in `realsense.py` are still valid. The real problems are **integration bugs between `composed_camera.py` and the RealSense driver, plus driver behavior that violates SONIC's wire format**. We must patch these before any smoke test will pass.

**Files audited:** `gear_sonic/camera/composed_camera.py`, `gear_sonic/camera/drivers/{realsense,oak,usb_camera,dummy}.py`, `gear_sonic/camera/sensor_server.py`, `gear_sonic/camera/sensor.py`, `gear_sonic/scripts/run_camera_viewer.py`, `gear_sonic/scripts/run_data_exporter.py`, `systemd/composed_camera_server.service`, `install_scripts/install_camera_server.sh`. Notion plan partially revised against ground truth.

### Wire-format facts (confirmed against code, supersede any Notion claim that disagrees)

- **Driver registry:** `composed_camera.py:_instantiate_camera` (lines 360–398). `--ego-view-camera realsense` is the right flag (matches Notion).
- **Image key per camera:** the driver's `mount_position` arg (e.g. `"ego_view"`) is the dict key under `images[...]`. The data exporter (`run_data_exporter.py:_add_images_to_frame_data`, line 532) demands `images["ego_view"]` — exact string.
- **Pixel format on the wire:** drivers must publish **RGB** numpy arrays (uint8, HxWx3). `oak.py:238` does `getCvFrame()[..., ::-1]` (BGR→RGB); `realsense.py` already requests `rs.format.rgb8`. Notion's claim "use BGR8" is **wrong** — leave RealSense as RGB8.
- **JPEG path:** `ImageMessageSchema.encode_image` (`sensor_server.py:220`) calls `cv2.imencode('.jpg', image, [JPEG_QUALITY, 80])`. Quality 80 is fixed; `--use-mjpeg` only affects OAK and is harmless if passed for RealSense. Notion's "do not pass `--use-mjpeg`" is good practice but not a hard error.
- **Camera-viewer convention:** `run_camera_viewer.py:111` does `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` for display — confirms RGB is the wire convention.
- **Publish rate vs capture rate:** `ComposedCameraConfig.fps` only controls the publish-loop sleep in `run_server`. The RealSense capture rate is hard-coded 30 Hz inside `RealSenseConfig.fps`. `--fps 30` matches.

### Concrete blockers in `gear_sonic/camera/drivers/realsense.py` and `composed_camera.py`

1. **BLOCKER — `device_id` is not forwarded to RealSense.** In `composed_camera._instantiate_camera`, line 379 reads:
   ```python
   return RealSenseSensor(mount_position=mount_position)
   ```
   The `device_id` parameter (passed in from `--ego-view-device-id`) is dropped on the floor. By contrast, OAK (line 373) and USB (line 393) both receive it. Result: with a single RealSense plugged in it works by accident; with multiple RealSense devices the driver always picks `devices[0]` after sorting by serial. Fix: pass `device_id=device_id` to the constructor and add a matching parameter to `RealSenseSensor.__init__`.
2. **BLOCKER — Depth stream is forced ON.** `realsense.py:64–78` enables both `rs.stream.color` and `rs.stream.depth`; `read()` returns `None` if either is missing (lines 100–105). This violates the "color only" hard constraint and risks USB-bandwidth contention.
3. **BLOCKER — Depth frames break the JPEG encoder.** `realsense.py:119–126` publishes an extra image key `f"{mount_position}_depth"` containing a uint16 depth array. `ImageMessageSchema.encode_image` then tries to `cv2.imencode('.jpg', depth_uint16, …)` — JPEG cannot encode uint16; this either raises in `composed_camera.run_server` or silently mangles bytes. Even if encode survives, `_add_images_to_frame_data` would error if the schema demanded `ego_view_depth` (it doesn't, so it's silently ignored on the recorder side — but the encode crash on the publisher side is fatal). Fix: don't enable the depth stream and don't add the `_depth` key.
4. **DESIGN BUG — `RealSenseSensor.__init__` signature uses `id: int = 0`.** The driver expects a 0-based index after sorting devices by serial, not a serial string. Fix: replace with `device_id: str | None = None`, and use `self.config.enable_device(device_id)` when provided, else just open the first available.
5. **Cosmetic — `RealSenseConfig` is a plain class with class-level annotations.** Same style as `OAKConfig`, so leaving as-is is fine — but if we add `enable_depth: bool = False` we should keep it consistent.

### Patch shape (Strategy A still viable, very small diff)

Two-file change, ~30 lines total:

- **`gear_sonic/camera/drivers/realsense.py`** — replace `id: int = 0` with `device_id: str | None = None`; gate depth on a default-off `enable_depth` flag; only publish the color frame; tidy `observation_space` to drop the depth space when disabled.
- **`gear_sonic/camera/composed_camera.py`** — line 376–379, forward `device_id` into `RealSenseSensor(...)`. This minimally edits a file outside `drivers/` but is justified by the Notion fallback clause "the driver-registry entry is genuinely missing" — the dispatch is genuinely incomplete.
- **`systemd/composed_camera_server.service`** — left untouched in the repo. The Orin-side unit gets generated dynamically by `install_camera_server.sh` (see P2.1 / P2.7 flow below), so we don't need to commit changes here unless we want to update the template's defaults.

This keeps Strategy A intact. Strategy C (custom thin publisher) is *not* needed unless the runtime smoke test exposes a `pyrealsense2` symbol break that we can't paper over — which would only show up after these patches land.

### Other corrections vs the Notion plan

- **`install_camera_server.sh` interactive flow (lines 156–202):** the script supports answering `realsense` to "Ego-view camera type" and entering a serial directly — no need to `Ctrl+C` past an OAK prompt the way Notion suggests. The script *does* attempt to start the service immediately after install, so we should answer **N** to "Install as systemd service?" on first run (in P2.1) and re-run / hand-edit the unit only after the foreground smoke test passes (P2.7).
- **Systemd template defaults:** `User=nvidia`, `HOME=/home/nvidia`. On our Orin the user is `unitree`; the install script substitutes `$USER` / `$HOME` automatically when generating the service, so the default values in the template don't matter as long as we let the install script regenerate.
- **`pyrealsense2` on aarch64:** Notion claims pre-built `manylinux2014_aarch64` wheels exist on JetPack 6.2. Validate at P2.3 — this can fail; if it does, fall back to building from source against the system `librealsense` (which is already installed because the decoupled stack runs RealSense). Add a fallback note to P2.3.

---

## Second-pass audit findings (2026-05-05, addendum) — what changed in our understanding

The first audit looked at the SONIC camera path in isolation. This pass found cross-cutting context that materially shapes the plan:

### F1. There is a known-working RealSense reference *in this repo* — `decoupled_wbc/control/sensor/realsense.py`
Same hardware, different package; the decoupled stack runs it daily. **Read-only reference, not editable.** Key facts vs the SONIC driver:
- **Depth defaults OFF** (`enable_depth: bool = False`) — exactly what we want SONIC to do.
- Uses `rs.format.bgr8` and then `bgr[..., ::-1]` to flip to RGB before publish. The SONIC driver uses `rs.format.rgb8` directly. Both produce the same RGB-on-the-wire result; **either is legal**, no need to change SONIC's choice.
- Has `serial_number: Optional[str] = None` plumbing — this is the pattern we're porting into SONIC in P2.X-2.
- Captures at **1280×720** and downsamples to **640×360** for the decoupled use case. **Do not copy this** — SONIC's `EGO_VIEW_HEIGHT=480, EGO_VIEW_WIDTH=640` (`gear_sonic/data/features_sonic_vla.py:18-19`); SONIC must emit `(480, 640, 3)`. Stick with the existing SONIC default of capturing 640×480 directly.
- Treats `bgra → np.asanyarray(...)` as the conversion path, no surprises.

Use it as a blueprint when writing the P2.X-2 patch. **Do not edit it.**

### F2. The "decoupled camera service" on the Orin is a shell script, not a systemd unit
- Repo root contains `start_camera_server.sh` which:
  1. `pkill -9 -f videohub_pc4` and `pkill -9 -f "realsense.*--server"` (a custom decoupled binary called `videohub_pc4` may hold the device — name to look for when diagnosing port-5555 conflicts).
  2. **Hardware-resets** the RealSense via `pyrealsense2`'s `device.hardware_reset()` + 4 s sleep — this is a real-world prereq for handing the camera between processes on the Orin.
  3. Runs `python3 -m decoupled_wbc.control.sensor.realsense --server --port 5555 --mount-position ego_view`.
- Notion (and our P0.3 / P2.7) talked about "disable the decoupled systemd service." There may not be one. The conflict-resolution story for P2.7 is more likely: (a) ensure `start_camera_server.sh` is not launched by anyone, (b) confirm no `videohub_pc4` / stray `realsense ... --server` process is alive, then (c) own port 5555 with our SONIC unit. **Operator should confirm whether their Orin actually has a systemd unit wrapping this script.**

### F3. The decoupled stack also ships a standalone publisher: `decoupled_wbc/scripts/realsense_zmq_server.py`
- ~150-line script, no `decoupled_wbc` imports beyond `pyrealsense2/zmq/msgpack/cv2/numpy`. Default port 5556.
- This is essentially **Strategy C already pre-implemented**. If our patched SONIC driver fails at runtime in some unfixable way, we can copy this script into `gear_sonic/camera/realsense_thin_publisher.py` (per Notion's fallback skeleton), retarget port to 5555, and use it instead of `composed_camera`. **Do not invoke unless P2.5/P2.6 reveals a hard blocker we can't patch; needs explicit user agreement per the constraint list.**

### F4. `gear_sonic[camera]` extra confirmed to *not* include `pyrealsense2`
- `gear_sonic/pyproject.toml` lines for `[project.optional-dependencies].camera` (verified): `pyzmq, msgpack, msgpack-numpy, opencv-python, tyro, depthai, requests`. No pyrealsense2. **P2.3's `pip install pyrealsense2` step is mandatory** — confirmed.

### F5. `launch_data_collection.py` auto-bootstraps via `_bootstrap_venv()`
- The script re-execs itself under `.venv_data_collection/bin/python` if `tyro` isn't importable. So Phase 3 doesn't strictly need `source .venv_data_collection/bin/activate` first. Worth noting when expanding Phase 3.
- Hard prerequisites the launcher checks (`launch_data_collection.py:_check_prerequisites`): `tmux`, `.venv_teleop`, `.venv_data_collection`, `gear_sonic_deploy/deploy.sh`, and (if `--sim`) `.venv_sim`. Since we're skipping sim, `.venv_sim` is not required.

### F6. Pane layout per the actual code matches Notion
- `launch_data_collection.py:_send_to_pane` calls confirm: pane 0 = C++ deploy (top-left), pane 1 = PICO teleop (bottom-left), pane 2 = data exporter (top-right, where the launcher lands you), pane 3 = camera viewer (bottom-right). Notion's description is correct. No correction needed.

### F7. Wrist-camera schema is present in features_sonic_vla.py
- `gear_sonic/data/features_sonic_vla.py:375-393` defines `observation.images.left_wrist` and `right_wrist` features. Activated by `--record-wrist-cameras` flag (data exporter line 100 + 917). Hard-constraint says we keep this off; confirmed it's a noop when off.

### F8. Workstation install script auto-`sudo apt install espeak`
- `install_data_collection.sh:18-22` runs `sudo apt-get install -y espeak` non-interactively (no prompt). It will silently warn and continue if it fails. **No sudo prompt to pause for in P1.1** unless the apt cache is stale and the install hangs on a password prompt — be alert but don't pre-emptively block.

## Status legend

- 🔲 TODO — not started
- 🟡 IN_PROGRESS — actively being worked on
- 🔴 BLOCKED — cannot proceed, see notes
- ✅ DONE — verified by acceptance check
- ⏸ DEFERRED — phase 4/5 stubs, expand after P3 lands

## Where this runs

- `[workstation]` — runs on `ethrc-thor@…ETHRC-Humanoid-WholeBodyControl` (this machine).
- `[orin]` — runs on the G1's Jetson Orin via `ssh unitree@192.168.123.164`. Agent SSHes; operator does not switch terminals.
- `[human-in-the-loop]` — physical action by Luca (powering hardware, donning PICO, pressing buttons, observing motion safety).

---

## Phase 0 — Prerequisites & sanity checks

### P0.1 🔲 TODO `[workstation]` — Confirm workstation repo identity
**Goal:** Verify the workstation repo at `~/ETHRC-Humanoid-WholeBodyControl` is the SONIC-capable fork and the deployment Quick Start has been completed previously.
**Steps:**
1. `pwd && git remote -v && git branch --show-current`
2. Confirm `gear_sonic/`, `gear_sonic_deploy/`, `decoupled_wbc/`, `install_scripts/install_data_collection.sh`, `install_scripts/install_camera_server.sh`, `systemd/composed_camera_server.service` all exist.
3. Note presence of `.venv_teleop` and `.venv_sim` (deployment Quick Start artifacts).
**Acceptance:** Repo matches expected layout; remote points at `ETHRoboticsClub/ETHRC-Humanoid-WholeBodyControl`; previous teleop venvs exist or user confirms Quick Start was done.
**Dependencies:** none.
**Notes:**
- (initial audit, 2026-05-05) Confirmed `gear_sonic/`, `gear_sonic_deploy/`, `decoupled_wbc/`, install scripts, and `systemd/composed_camera_server.service` all present. Remote is `ETHRoboticsClub/ETHRC-Humanoid-WholeBodyControl`, current branch `feat/sonic-data-collection`. Still need to verify `.venv_teleop` / `.venv_sim`.

### P0.2 🔲 TODO `[human-in-the-loop]` — Confirm SONIC checkpoints + PICO + tight clothing
**Goal:** Verify physical and asset prerequisites: SONIC HF checkpoints downloaded, PICO calibrated with XRoboToolkit, tight-fitting pants ready for the operator.
**Steps:**
1. `[HUMAN]` Confirm the workstation has the `nvidia/GEAR-SONIC` checkpoints (typically under `motionbricks/` or wherever `download_from_hf.py` placed them).
2. `[HUMAN]` Confirm PICO 4 Ultra Enterprise is charged, calibrated, has XRoboToolkit installed, and pairs with the workstation's `XRoboToolkit-PC-Service`.
3. `[HUMAN]` Confirm tight-fitting pants/leggings are physically on hand (mandatory for foot trackers in POSE mode).
**Acceptance:** Luca confirms all three.
**Dependencies:** none.
**Notes:**

### P0.3 🔲 TODO `[orin]` `[human-in-the-loop]` — Confirm decoupled RealSense server healthy + map current launch mechanism
**Goal:** Confirm USB/udev/firmware path for the RealSense is healthy on the Orin by verifying the existing decoupled-WBC camera server runs. **Also map exactly how it's launched on the operator's Orin** (manual shell script vs systemd unit vs custom autostart) — the conflict-resolution story in P2.7 depends on this. **Operator handles network bring-up themselves; this plan does not run network-connectivity checks.**
**Prereq (operator-managed, not tracked here):** Orin powered on and reachable from the workstation over `192.168.123.x`.
**Steps:**
1. `ssh unitree@192.168.123.164` (operator confirms connectivity is up before this sub-task starts).
2. Audit how the decoupled camera server is currently launched on this Orin:
   - `systemctl list-units --type=service | grep -i 'camera\|realsense'` — any systemd unit?
   - `crontab -l` — anything in cron?
   - `ls /etc/systemd/system/ | grep -i 'camera\|realsense'`
   - Check `~/<repo>/start_camera_server.sh` exists (matches the workstation copy at repo root) and confirm whether it is invoked manually, from a launcher script, or from a unit file.
3. Run the decoupled server once to validate hardware (per the operator's normal procedure — typically `bash ~/<repo>/start_camera_server.sh`). Confirm frames flow on its port.
4. Record (in notes below): launch mechanism, service/unit name (if any), port used, any process that holds the device (`videohub_pc4` per the script's `pkill` line).
**Acceptance:** Operator confirms decoupled server starts and produces frames; the launch mechanism + port is recorded below.
**Dependencies:** none.
**Notes:**
- Decoupled launch mechanism: _<TBD — manual script / systemd / cron>_
- Port used by decoupled server: _<TBD>_
- Other holder processes seen (e.g. `videohub_pc4`): _<TBD>_

### P0.4 🔲 TODO `[workstation]` — Disk sanity
**Goal:** Confirm enough disk for `.venv_data_collection` (~few GB with PyAV/OpenCV/LeRobot).
**Steps:**
1. `df -h ~` — confirm >10 GB free.
**Acceptance:** Disk OK.
**Dependencies:** none.
**Notes:**

---

## Phase 1 — Workstation one-time install

### P1.1 🔲 TODO `[workstation]` — Run `install_data_collection.sh`
**Goal:** Create the third venv `.venv_data_collection` with `gear_sonic[data_collection]` deps (LeRobot, PyAV, OpenCV, espeak system pkg).
**Steps:**
1. `cd ~/ETHRC-Humanoid-WholeBodyControl`
2. `bash install_scripts/install_data_collection.sh`
3. The script calls `apt install espeak` — that needs `sudo`. **Pause and confirm with user before approving the sudo prompt.**
**Acceptance:** Exit code 0; `.venv_data_collection/` exists.
**Dependencies:** P0.1.
**Notes:**

### P1.2 🔲 TODO `[workstation]` — Verify `.venv_data_collection` imports
**Goal:** Smoke-import the heavy deps so we know the env is functional before relying on it for camera viewer + dataset recorder.
**Steps:**
1. `source .venv_data_collection/bin/activate`
2. `python -c "import lerobot, av, cv2; print('ok')"`
**Acceptance:** Prints `ok`, no traceback.
**Dependencies:** P1.1.
**Notes:**

---

## Phase 2 — Orin RealSense camera server install (Strategy A) — **MAIN WORK**

### P2.0 🔲 TODO `[workstation]` — Create branch `feat/realsense-camera-sonic` from `feat/sonic-data-collection`
**Goal:** Isolate driver work on a dedicated branch off the current SONIC integration branch, so the diff is reviewable and easy to upstream.
**Steps:**
1. `cd ~/ETHRC-Humanoid-WholeBodyControl`
2. `git status` (must be clean before branching).
3. `git checkout feat/sonic-data-collection` (confirm we're branching from the right base).
4. `git checkout -b feat/realsense-camera-sonic`
**Acceptance:** `git branch --show-current` returns `feat/realsense-camera-sonic`; `git merge-base --is-ancestor feat/sonic-data-collection HEAD` succeeds; tree clean.
**Dependencies:** P0.1.
**Notes:**

### P2.1 🔲 TODO `[orin]` — Update repo on Orin + run `install_camera_server.sh` (skip systemd this pass)
**Goal:** Bring the Orin repo to the branch carrying our driver patches (`feat/realsense-camera-sonic`), then stand up `.venv_camera`. We want the install script's interactive flow to succeed without auto-starting a broken service, so we **answer `N` to "Install as systemd service?"** and defer that to P2.7.
**Steps:**
1. `ssh unitree@192.168.123.164`
2. Locate the existing repo: `ls ~ && cd <repo>` and record path. Record current state with `git status && git branch --show-current && git remote -v`.
3. **Pause and confirm with user** before switching branches. Once approved, fetch and check out the patched branch:
   - `git fetch <remote>`
   - `git checkout feat/realsense-camera-sonic && git pull --ff-only` (this branch will only exist on the remote *after* P2.X-1 / P2.X-2 patches are pushed; if those aren't pushed yet, hold this sub-task until they are).
4. `bash install_scripts/install_camera_server.sh` — interactive flow:
   - It will run `detect_oak_cameras`, find none, print "(no OAK devices detected)", and prompt to retry. Answer `N` to skip.
   - At "Ego-view camera type [oak]:" answer `realsense`.
   - At "Ego-view device ID (MxID or /dev/video index):" leave blank (we'll fill the serial in P2.7) — or answer with the serial from P2.4 if already known.
   - Decline left-wrist and right-wrist prompts (Enter / `n`).
   - Accept default port `5555`.
   - At "Install the camera server as a systemd service (auto-start on boot)?" answer **`N`**. We'll do this in P2.7 after the foreground smoke test passes.
**Acceptance:** `.venv_camera/` exists on the Orin; `<repo_path>/.venv_camera/bin/python -c "import depthai, zmq, msgpack, cv2, tyro; print('ok')"` returns ok. No systemd service was installed yet.
**Dependencies:** P0.3, P2.X-1, P2.X-2 (driver patches must be on a fetchable remote branch first).
**Notes:**
- Orin repo path: _<TBD — fill in during step 2>_
- Orin branch before / after update: _<TBD>_

### P2.2 ✅ DONE `[workstation]` — Audit bundled `realsense.py` + `composed_camera.py`
**Goal:** Before installing or running anything else, read what's actually shipped. Findings drive P2.X patches.
**Acceptance:** All audit questions answered; concrete blocker list produced.
**Dependencies:** none.
**Notes:**
- Done 2026-05-05 via direct read of all camera-server source files. Findings consolidated in the **"Camera-server audit findings"** section near the top of this file. TL;DR:
  - `--ego-view-camera realsense` is the correct dispatcher value (composed_camera.py:375).
  - `--ego-view-device-id <serial>` is the intended format **but it is currently dropped on the floor** for RealSense (composed_camera.py:379 hardcodes the constructor call); driver currently uses an integer index `id` after sorting devices by serial. → see **P2.X-1** patch.
  - Driver enables both color and depth streams, publishes a `<mount>_depth` uint16 array, and the JPEG encoder cannot encode uint16 — depth is a hard blocker, not just bandwidth. → **P2.X-2** patch.
  - Color format is `rs.format.rgb8`, which matches the wire convention used by OAK (RGB on the wire). Notion's "force BGR8" guidance is wrong; do not change this.
  - 640×480 @ 30 fps is already the driver default; no change needed.
  - No deprecated `pyrealsense2` symbols spotted.
  - Systemd unit template ships with placeholder `nvidia` user; install script regenerates from `$USER`/`$HOME` so we don't have to commit a change there.

### P2.3 🔲 TODO `[orin]` — Install `pyrealsense2` in `.venv_camera` + verify device visible
**Goal:** Get `pyrealsense2` into `.venv_camera` and confirm the SDK enumerates the camera at the OS level.
**Steps:**
1. `ssh unitree@192.168.123.164`
2. `cd <repo_path_from_P2.1> && source .venv_camera/bin/activate`
3. `pip install pyrealsense2`  *(no sudo)*. If pip fails to find a wheel for aarch64 / Python 3.10:
   - Check whether `pyrealsense2` is already installed system-wide on the Orin (the decoupled stack uses RealSense, so `librealsense` and bindings are likely present): `python -c "import sys; sys.path.insert(0,'/usr/lib/python3/dist-packages'); import pyrealsense2; print(pyrealsense2.__file__)"`. If yes, recreate `.venv_camera` with `--system-site-packages` (re-run install_camera_server.sh after deleting the venv) — **pause and confirm with user before destroying the venv.**
   - As a last resort, build pyrealsense2 from source against the system librealsense (this is heavyweight; ask user before going there).
4. `python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])"`
5. If list is empty: check `dmesg | grep -i realsense`, USB 3 cable, and udev rules at `/etc/udev/rules.d/99-realsense-libusb.rules`. (Decoupled stack already worked here, so udev rules are almost certainly already in place — confirm before touching.) **Installing udev rules requires sudo — pause and ask user.**
**Acceptance:** Step 4 prints a non-empty list, e.g. `['Intel RealSense D435i']`.
**Dependencies:** P2.1.
**Notes:**

### P2.4 🔲 TODO `[orin]` — Identify camera serial number
**Goal:** Capture the exact serial we'll pass to `--ego-view-device-id` and burn into the systemd unit.
**Steps:**
1. `python -c "import pyrealsense2 as rs; ctx=rs.context(); [print(d.get_info(rs.camera_info.serial_number), '|', d.get_info(rs.camera_info.name)) for d in ctx.devices]"`
2. Record the serial in this notes section.
**Acceptance:** Serial recorded below.
**Dependencies:** P2.3.
**Notes:**
- Serial: _<TBD — fill in after running>_

### P2.X-1 🔲 TODO `[workstation]` — Patch `composed_camera.py` factory to forward `device_id` to RealSense
**Goal:** Fix the dropped-`device_id` integration bug so `--ego-view-device-id <SERIAL>` actually selects a specific RealSense camera.
**Steps (all in `gear_sonic/camera/composed_camera.py`, branch `feat/realsense-camera-sonic`):**
1. In `_instantiate_camera`, change the RealSense branch from:
   ```python
   return RealSenseSensor(mount_position=mount_position)
   ```
   to:
   ```python
   return RealSenseSensor(mount_position=mount_position, device_id=device_id)
   ```
2. Run lint / type-check locally.
**Acceptance:** Diff is exactly one logical line in `_instantiate_camera`; nothing else in `composed_camera.py` is modified; `python -m compileall gear_sonic/camera/composed_camera.py` succeeds.
**Dependencies:** P2.0.
**Notes:**

### P2.X-2 🔲 TODO `[workstation]` — Patch `realsense.py` (device_id arg + color-only stream)
**Goal:** Make the RealSense driver (a) accept a `device_id: str | None` to match the new factory call, (b) default to color-only (no depth), (c) only publish the `ego_view` color image — no `_depth` key, no uint16 going into the JPEG encoder.
**Reference (read-only):** `decoupled_wbc/control/sensor/realsense.py` already implements this pattern correctly (`enable_depth: bool = False` default, `serial_number: Optional[str] = None`, single-key publish path). Use it as a blueprint — but **keep capture at 640×480 directly**; the decoupled driver captures 1280×720 and downsamples to 640×360, which is incompatible with SONIC's `(480, 640, 3)` shape requirement.
**Steps (all in `gear_sonic/camera/drivers/realsense.py`):**
1. Add `enable_depth: bool = False` to `RealSenseConfig` (default off).
2. Replace the `__init__` parameter `id: int = 0` with `device_id: str | None = None`. When provided, call `self.config.enable_device(device_id)`. When None, fall through to opening the first available device (drop the sort-and-index logic; just rely on the SDK's default selection or pick `devices[0]` after enumeration without indexing by `id`).
3. Inside `__init__`, only call `self.config.enable_stream(rs.stream.depth, …)` when `config.enable_depth` is True.
4. In `read()`, only fetch / extract / return the depth frame when `self._realsense_config.enable_depth` is True. Default path: return `{"timestamps": {mount_position: t}, "images": {mount_position: color_rgb}}` — exactly one key.
5. Update `observation_space()` to omit the depth space when depth is disabled.
6. Leave `rs.format.rgb8` as-is — both `rs.format.bgr8 + bgr[..., ::-1]` (decoupled style) and `rs.format.rgb8` direct (current SONIC style) produce identical RGB-on-the-wire output. Don't churn this for stylistic alignment.
7. Add a top-of-file note dating the patch and citing JetPack 6.2 / pyrealsense2 version range tested.
**Acceptance:** `python -m compileall gear_sonic/camera/drivers/realsense.py` succeeds; with the new code, calling `RealSenseSensor(mount_position="ego_view")` (no device_id, no depth) succeeds against a single connected RealSense and `read()` returns a dict whose `images` has exactly one key, `"ego_view"`, with shape `(480, 640, 3)` and dtype `uint8`.
**Dependencies:** P2.0, P2.X-1 (the constructor signatures must agree before either is committed).
**Notes:**

### P2.X-3 🔲 TODO `[workstation]` — Commit + push `feat/realsense-camera-sonic` so the Orin can pull
**Goal:** Get the patches onto a remote branch so `git pull` on the Orin (P2.1 step 3) can fetch them. **Do not push until user reviews the diff.**
**Steps:**
1. `git diff feat/sonic-data-collection..feat/realsense-camera-sonic` — share the diff with the user for review.
2. `git status` clean except for the two patched files.
3. After explicit user approval: `git push -u origin feat/realsense-camera-sonic`.
**Acceptance:** User has reviewed and approved the diff; remote branch exists.
**Dependencies:** P2.X-1, P2.X-2.
**Notes:**

### P2.5 🔲 TODO `[orin]` — Manual smoke test of the camera server
**Goal:** Run `composed_camera` interactively against RealSense so any remaining errors surface in the foreground.
**Pre-flight checks (do these *before* starting the server):**
- Confirm no other process holds the RealSense. Mirror the cleanup the decoupled `start_camera_server.sh` does:
  ```
  pgrep -af "videohub_pc4|realsense.*--server|composed_camera" || echo "no holders"
  ```
  If holders exist, stop them per P0.3 findings (don't `pkill -9` reflexively — confirm with user first).
- Optional but recommended on a "warm" Orin (camera was just used by the decoupled stack): hardware-reset the device so SONIC can claim it cleanly:
  ```
  python -c "import pyrealsense2 as rs, time; [d.hardware_reset() for d in rs.context().query_devices()]; time.sleep(4)"
  ```
**Steps:**
1. `source .venv_camera/bin/activate`
2. ```
   python -m gear_sonic.camera.composed_camera \
       --ego-view-camera realsense \
       --ego-view-device-id <SERIAL_FROM_P2.4> \
       --port 5555 \
       --fps 30
   ```
3. Watch the foreground log for: `Initializing RealSense sensor for camera type: realsense`, `Done initializing RealSense sensor: <serial>`, periodic `Image sending FPS: ~30.00`, and `[Sensor server] Message sent: …`. No traceback.
4. If a new failure surfaces (post-patches), classify against the Notion decision tree and either spawn a new P2.X-N sub-task or escalate to user. Do not silently retry.
5. Leave the process running (foreground) for P2.6.
**Acceptance:**
- Server stays up for ≥60 s,
- prints `Image sending FPS: ~30.00` at ~10 s intervals,
- no `cv2.imencode` errors (proves color-only path),
- no `RuntimeError: No device connected` / `Couldn't resolve requests`.
**Dependencies:** P2.X-1, P2.X-2, P2.X-3 (patches landed on Orin), P2.3 (sdk), P2.4 (serial).
**Notes:**

### P2.6 🔲 TODO `[workstation]` `[orin]` — Verify wire format from workstation
**Goal:** While P2.5's foreground server is still running on the Orin, confirm the workstation's camera viewer subscribes successfully — proves the driver speaks SONIC's protocol end-to-end (msgpack envelope, RGB, JPEG-quality 80, single `ego_view` image key).
**Steps (workstation):**
1. `cd ~/ETHRC-Humanoid-WholeBodyControl && source .venv_data_collection/bin/activate`
2. ```
   python gear_sonic/scripts/run_camera_viewer.py \
       --camera-host 192.168.123.164 --camera-port 5555
   ```
3. The viewer prints `Detected N camera stream(s): ['ego_view']`. **Confirm N == 1 and the only key is `ego_view`** — if it logs `ego_view_depth` too, P2.X-2 wasn't applied correctly.
4. Visually confirm: tiled OpenCV window opens, RealSense feed at 640×480, ~30 Hz, colors look correct (no R/B swap), no corruption.
**Acceptance:** Live RealSense feed visible at 640×480; viewer reports exactly one stream named `ego_view`; latency log lines appear (`Image latency for ego_view: ~XX ms`).
**Dependencies:** P1.2 (workstation venv), P2.5 (Orin server foreground).
**Notes:**

### P2.7 🔲 TODO `[orin]` `[human-in-the-loop]` — Generate systemd unit, disable conflicting service, reboot test
**Goal:** Autostart the SONIC camera server on Orin boot. Disable the decoupled-WBC camera server (or move it off port 5555) to prevent dual-bind. Reboot and verify autostart.
**Two install options (pick one with user):**
- **Option A (preferred): re-run `install_camera_server.sh`** non-destructively — re-running it deletes and rebuilds `.venv_camera`, which is wasteful but produces a correct unit. *Skip this option.*
- **Option B (recommended): hand-write the unit** using the install-script template as reference, since the venv already exists from P2.1.
**Steps (Option B):**
1. `[HUMAN]` Confirm sudo on Orin and approve unit installation + reboot.
2. Capture serial and repo path:
   - `SERIAL=<from-P2.4>`, `REPO=<from-P2.1>`, `USER=unitree`, `HOME=/home/unitree`.
3. `[HUMAN]` Approve sudo. On Orin write `/etc/systemd/system/composed_camera_server.service` with content (taken from `install_camera_server.sh` lines 209–228, substituted):
   ```
   [Unit]
   Description=SONIC Composed Camera Server (ZMQ)
   After=network.target

   [Service]
   Type=simple
   User=unitree
   Environment="HOME=/home/unitree"
   Environment="REPO_DIR=<REPO>"
   WorkingDirectory=<REPO>
   ExecStart=<REPO>/.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --ego-view-device-id <SERIAL> --port 5555 --fps 30
   Restart=on-failure
   RestartSec=5
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
4. Resolve port-5555 conflict with the decoupled stack (use the launch-mechanism findings from P0.3):
   - **If decoupled launch is a systemd unit:** `sudo systemctl disable --now <unit-name>`.
   - **If decoupled launch is the manual `start_camera_server.sh`:** there's nothing to disable — just confirm no operator workflow auto-starts it on boot, and that `videohub_pc4` / stale `python ... realsense ... --server` processes are not running (`pgrep -af videohub_pc4` and the `realsense.*--server` pattern from `start_camera_server.sh:24`).
   - As a last resort, change one side off port 5555 (decoupled side, since SONIC's modality.json + workstation defaults assume 5555).
   - **Pause and confirm with user which approach.**
5. `sudo systemctl daemon-reload && sudo systemctl enable --now composed_camera_server.service && sudo systemctl status composed_camera_server.service`.
6. `[HUMAN]` `sudo reboot` the Orin.
7. After reboot: `ssh unitree@192.168.123.164 "sudo systemctl status composed_camera_server.service"` — confirm `Active: active (running)`.
8. Re-run the workstation camera viewer (P2.6 command) post-reboot to confirm frames flow.
**Acceptance:** Service `Active: active (running)` after reboot; workstation viewer shows live feed; no port conflict in journal logs.
**Dependencies:** P2.6, P0.3 (decoupled service name).
**Notes:**

---

> **Sim2sim verification (Notion Phase 3) is intentionally skipped.** Operator will validate the SMPL → policy → exporter chain directly on real hardware as part of Phase 3 below. Re-introduce a sim phase only if a real-hardware bring-up failure suggests we need to bisect the chain in MuJoCo.

## Phase 3 — Real-robot recording session ⏸ DEFERRED

⏸ One-line stub. Expand after Phase 2 lands. Per Notion (their "Phase 4"): hardware bring-up, `launch_data_collection.py` with `--camera-host 192.168.123.164`, engage POSE on PICO, record episodes via **Left Grip + A**, end via **A+X** then **A+B+X+Y**. Lives in Notion section "Phase 4 — Real-robot recording session" (Steps 4.1–4.6).
**Confirmed-from-code prereqs** (will lift into sub-tasks when expanding):
- `.venv_teleop`, `.venv_data_collection`, `gear_sonic_deploy/deploy.sh`, and `tmux` must exist on the workstation (`launch_data_collection.py:_check_prerequisites`).
- The launcher auto-bootstraps to `.venv_data_collection/bin/python` via `_bootstrap_venv()`; no need to `source` first.
- 4-pane layout (verified in code): pane 0 = C++ deploy, pane 1 = PICO teleop, pane 2 = data exporter (where the launcher lands you), pane 3 = camera viewer.
- Workstation NIC must be set to `192.168.123.222/24` and route to `192.168.123.164` for the camera client + state subscriber to connect.

## Phase 4 — Post-processing & merging ⏸ DEFERRED

⏸ One-line stub. Expand after Phase 3 lands. Per Notion (their "Phase 5"): `process_dataset.py` cleans stale-SMPL frames + frozen lead-ins; supports merging multiple sessions via `--dataset-list`. Lives in Notion section "Phase 5 — Post-processing & merging".

---

## Working-loop reminders (for future-me / future-session)

1. Read this file at the start of every step.
2. Pick the next 🔲 TODO with no unmet deps; flip to 🟡 IN_PROGRESS.
3. Execute → run acceptance check → ✅ DONE on success, 🔴 BLOCKED on failure.
4. Append findings to the sub-task's notes section, dated, terse.
5. Stop after every sub-task and summarise to the user. Wait for go-ahead.
6. Hard constraints from the user: never push to remote without diff review; never run sudo on Orin without explicit confirmation; do not pursue Strategy B; do not edit anything in the don't-touch list.
