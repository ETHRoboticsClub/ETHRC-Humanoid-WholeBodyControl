# Decoupled WBC on G1 — ETHRC Setup & Teleop Recording Guide

End-to-end instructions for installing the **Decoupled WBC** Docker stack on a workstation, teleoperating the **Unitree G1** with a **Pico** headset, and recording task data.

Authoritative upstream documentation: <https://nvlabs.github.io/GR00T-WholeBodyControl/references/decoupled_wbc.html>
ETHRC repo: this repository (`ETHRC-Humanoid-WholeBodyControl/`).

> Everything below is run from the repo root unless noted.

---

## 0. Step-by-step Plan (TL;DR)

1. Verify host prerequisites (Ubuntu, NVIDIA driver, Docker, NVIDIA Container Toolkit, git-lfs).
2. Make sure the repo is cloned with Git LFS contents.
3. Bump the joint safety limits (one-time edit on a fresh clone).
4. Install the Decoupled WBC Docker image (`run_docker.sh --install --root`).
5. Configure the ethernet link to the G1 (static IP `192.168.123.222/24`).
6. Power on the G1 and put it in Developer Mode.
7. Bring up the Pico headset on the same LAN as the workstation + launch `XRoboToolkit-PC-Service`.
8. SSH to the G1 and launch the camera server (`./start_camera_server.sh`).
9. Enter the Decoupled WBC Docker container (`run_docker.sh --root`).
10. Run `deploy_g1.py` (one-shot tmux deploy) OR the 3 individual commands.
11. Drive teleop with Pico, record episodes with controller `A`, discard with `B`.

Each step is detailed below.

---

## 1. Host Prerequisites

Upstream supports **Ubuntu 22.04**. This workstation is currently on Ubuntu 24.04 — the Docker image still works (everything runs inside the container) but if you hit driver/runtime weirdness, a 22.04 host is the supported baseline.

Required on the host:

- Ubuntu 22.04 (recommended) — 24.04 should work.
- NVIDIA driver compatible with your GPU. Verify with:
  ```bash
  nvidia-smi
  ```
- Docker Engine.
  ```bash
  docker --version
  ```
- **NVIDIA Container Toolkit** (gives the container GPU access). Install once:
  ```bash
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
  > `decoupled_wbc/docker/run_docker.sh` will perform this install automatically the first time you run `--build`, but installing it once up-front is cleaner.
- Git + Git LFS:
  ```bash
  sudo apt update && sudo apt install -y git git-lfs
  git lfs install
  ```

> **Tip:** add yourself to the `docker` group to avoid typing `sudo` for every docker command:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker     # or log out / log back in
> ```
> The repo's `run_docker.sh` still uses `sudo docker …` internally, so this is a convenience only.

---

## 2. Repository Setup

If you don't already have the repo:

```bash
git clone https://github.com/ETHRoboticsClub/ETHRC-Humanoid-WholeBodyControl.git
cd ETHRC-Humanoid-WholeBodyControl
git lfs pull
```

LFS pulls the ONNX policies and meshes. Without `git lfs pull` you'll see small pointer files and the control loop will silently fail.

Optional: verify the host-side environment expectations
```bash
python check_environment.py
```

---

## 3. Safety Limits (one-time, fresh clone)

Open `decoupled_wbc/control/envs/g1/utils/joint_safety.py` and confirm the velocity limits at the top of `JointSafetyMonitor`:

| Parameter             | Default upstream | Required for teleop |
|-----------------------|------------------|---------------------|
| `ARM_VELOCITY_LIMIT`  | 6.0 rad/s        | **10.0 rad/s**      |
| `HAND_VELOCITY_LIMIT` | 50.0 rad/s       | **70.0 rad/s**      |

In this repo they are already set to the teleop-friendly values (10.0 / 70.0). If you ever re-pull upstream defaults, bump them back — otherwise the safety monitor trips and locks the robot during fast teleop motions.

---

## 4. Install the Docker Image

All control stack commands run **inside** the Decoupled WBC Docker container. The image bundles all Python deps (Pico SDK, MuJoCo, RealSense, ZMQ, etc.).

From the repo root:

```bash
cd decoupled_wbc
./docker/run_docker.sh --install --root
```

What this does (see `docker/run_docker.sh`):

- Pulls `nvgear/gr00t_wbc:latest` from Docker Hub.
- Tags it locally as `decoupled_wbc-deploy-root` (or `…-$USER` if you drop `--root`).
- Removes any old containers with the same name to keep the install clean.

The pull is a few GB — do it on a fast connection the first time.

If you want to build the image locally instead (e.g. you don't want to run as root inside the container):

```bash
./docker/run_docker.sh --build
```

This builds from `docker/Dockerfile.deploy`, installs the NVIDIA Container Toolkit if missing, and tags the image under your username.

To enter / re-enter a shell in the container:

```bash
./docker/run_docker.sh --root
```

The first time, this creates a long-lived bash container named `decoupled_wbc-bash-root`. Subsequent runs `docker exec` into that container so your state (history, installed extras) persists.

Useful flags:

| Flag        | Effect                                                                 |
|-------------|------------------------------------------------------------------------|
| `--install` | Pull prebuilt image from Docker Hub.                                   |
| `--build`   | Build locally as non-root user.                                        |
| `--root`    | Run/install as root inside the container.                              |
| `--branch`  | Branch-scoped container name (lets you have one container per branch). |
| `--deploy`  | Run `decoupled_wbc/scripts/deploy_g1.py` directly inside a fresh container instead of dropping to a shell. |
| `--clean`   | Stop and remove the deploy/bash containers.                            |

The container mounts the repo at `~/Projects/<repo-name>` and sets `PYTHONPATH` so any edit you make on the host is reflected immediately inside.

---

## 5. Network Setup (Real Robot)

Plug an ethernet cable from your workstation to the G1's developer port, then put your NIC on the G1 subnet. The G1's fixed address is `192.168.123.164/24`.

```bash
# Replace enp4s0 with your interface (check `ip -br addr`).
sudo ip addr add 192.168.123.222/24 dev enp4s0
sudo ip link set enp4s0 up
ping 192.168.123.164
```

> On this workstation the interface is `enp4s0` and the static IP `192.168.123.222/24` is already configured (verify with `ip -br addr`).

If `ping` fails:
- Ethernet cable disconnected? `ip link show enp4s0` will show `NO-CARRIER`.
- G1 not in Developer Mode? Press `L2+R2`, `L2+A`, `L2+B` on the physical controller — robot announces *"Developer mode"*.
- IP already used by another interface? `sudo ip addr flush dev enp4s0` and re-add.

For WiFi-based G1 connection, see <https://huggingface.co/docs/lerobot/en/unitree_g1>.

---

## 6. Pico Headset Setup

The Pico runs the XR Robotics teleop app; the PC runs the matching service. The Docker image already contains the Python bindings used by `pico_streamer.py`. You only need the desktop service:

- **XRoboToolkit-PC-Service** — <https://github.com/XR-Robotics/XRoboToolkit-PC-Service>

Install per its README on the workstation (outside Docker — it's a separate companion service that the streamer connects to over localhost / LAN).

Pre-session checks:

1. Headset + PC on the same LAN. The easiest reliable setup is a phone hotspot, both devices connected to it.
2. Launch `XRoboToolkit-PC-Service` on the PC, put on the headset, launch the teleop app — both should report "Connected".
3. Unit-test the connection from inside the Decoupled WBC container:
   ```bash
   python decoupled_wbc/control/teleop/streamers/pico_streamer.py
   ```
   You should see controller poses streaming.

### Pico controller bindings

| Input                  | Action                                  |
|------------------------|-----------------------------------------|
| `menu + left trigger`  | Toggle lower-body policy                |
| `menu + right trigger` | Toggle upper-body policy                |
| Left stick             | X/Y translation                         |
| Right stick            | Yaw rotation                            |
| L / R triggers         | Hand gripper control                    |
| `A` (during teleop)    | Start / stop episode recording          |
| `B` (during teleop)    | Discard current trajectory              |

---

## 7. Start the Camera Server on the G1

SSH into the G1 (e.g. `ssh unitree@192.168.123.164`) and start the generic composed camera server. The script lives at the repo root:

```bash
./start_camera_server.sh \
  --ego-camera zed \
  --left-camera oak --left-device-id <LEFT_OAK_MXID> \
  --right-camera oak --right-device-id <RIGHT_OAK_MXID> \
  --port 5555
```

If you only need a single ego camera, run:
```bash
./start_camera_server.sh --ego-camera realsense --port 5555
```

The script supports `realsense`, `zed`, `oak`, `oak_mono`, and `usb` camera types, with optional left/right/head wrist cameras. When `realsense` is selected, the launcher will also perform a hardware reset before starting the server.

Leave it running. The workstation will publish/subscribe to it over ZMQ during teleop.

---

## 8. Start the Decoupled WBC Stack

You have **two equivalent ways** to bring up the control + teleop + data exporter trio.

### Option A — One-shot deploy (recommended)

Enter the container, then run:

```bash
./decoupled_wbc/docker/run_docker.sh --root          # drop into the container

# inside the container:
python decoupled_wbc/scripts/deploy_g1.py \
    --interface real \
    --hand_control_device pico \
    --body_control_device pico \
    --camera_host 192.168.123.164 \
    --no-add-stereo-camera \
    --record-wrist-cameras
```

`deploy_g1.py` spawns a tmux session `g1_deployment` with three panes:

| Pane                            | Process                                |
|---------------------------------|----------------------------------------|
| `0.0` (left)                    | `run_g1_control_loop.py`               |
| `0.1` (top right)               | `run_g1_data_exporter.py`              |
| `0.2` (bottom right)            | `run_teleop_policy_loop.py` (Pico)     |

Useful tmux keys: `Ctrl+b` then `d` to detach, `Ctrl+\` in any pane to kill the session.

### Option B — Three terminals

Open three terminals **inside the container** (`./decoupled_wbc/docker/run_docker.sh --root` from each — they all attach to the same bash container) and run one command per terminal:

**Terminal 1 — control loop**
```bash
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real
```

**Terminal 2 — teleop policy (Pico)**
```bash
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --hand_control_device=pico \
    --body_control_device=pico
```

**Terminal 3 — data exporter**
```bash
python decoupled_wbc/control/main/teleop/run_g1_data_exporter.py \
  --camera-host 192.168.123.164 \
  --no-add-stereo-camera \
  --record-wrist-cameras
```

> If you're not connected to the real robot yet and just want to validate the pipeline in MuJoCo, use `--interface sim` on the control loop (and `--camera_host localhost` / appropriate sim camera config on the exporter).

---

## 9. Recording a Teleop Episode

Once all three processes are up:

1. In the `control_data_teleop` pane (left), press **`]`** to activate the lower-body policy. The robot will hold its balanced pose.
2. Put on the Pico, hold a comfortable T-pose, then press **`menu + right trigger`** to engage the upper-body policy (and `menu + left trigger` if you also want locomotion). The robot now mirrors the headset/controllers.
3. Press **`A`** on the Pico controller to **start recording**. The data exporter writes the episode (state, action, video frames) to disk.
4. Perform the task.
5. Press **`A`** again to **stop recording**.
6. Don't like the take? Press **`B`** instead to **discard** the trajectory.
7. To safely abort: press **`o`** in the control pane to deactivate the policy; **`9`** releases / holds the robot.

Episode files land under the data exporter's output root (see `--root_output_dir` in `decoupled_wbc/control/main/teleop/configs/configs.py` — the default is wired through `DeploymentConfig`). Tip: pass `--root_output_dir /path/of/your/choice` to either `deploy_g1.py` or `run_g1_data_exporter.py` to keep data organized per task.

### Keyboard shortcuts (control pane)

| Key         | Action                                  |
|-------------|-----------------------------------------|
| `]`         | Activate policy                         |
| `o`         | Deactivate policy                       |
| `9`         | Release / hold robot                    |
| `k`         | Reset simulation and policies           |
| `w`/`s`     | Forward / backward                      |
| `a`/`d`     | Strafe left / right                     |
| `q`/`e`     | Rotate left / right                     |
| `z`         | Zero navigation commands                |
| `1`/`2`     | Raise / lower base height               |
| `backspace` | Reset robot in visualizer               |
| `` ` ``     | Terminate the tmux session             |
| `Ctrl+D`    | Exit the shell in the pane              |

---

## 10. Every-Session Checklist (real G1)

Once everything is installed, the daily flow is:

1. **Power on the G1**, enter Developer Mode: `L2+R2`, `L2+A`, `L2+B`.
2. **Plug ethernet** to the workstation.
3. **Verify connectivity**: `ping 192.168.123.164` (set up workstation IP per §5 if needed).
4. **Pico** on the same LAN as the PC; launch the teleop app on the headset and `XRoboToolkit-PC-Service` on the PC.
5. **SSH the G1** and run `./start_camera_server.sh`.
6. **Enter Docker** on the workstation: `./decoupled_wbc/docker/run_docker.sh --root`.
7. **Deploy** with the `deploy_g1.py` command (or three-terminal Option B).
8. **Record** episodes with Pico `A` / discard with `B`.

---

## 11. Troubleshooting

**`nvidia-container-toolkit` missing inside the run.**
Install per §1 and `sudo systemctl restart docker`. Then re-run `--install`.

**`PermissionError: [Errno 13] Permission denied` on host files after running `--root`.**
Some files in the repo are now root-owned (mounted volume + root inside container). Fix on host:
```bash
sudo chown -R $USER:$USER .
```
Run with `--build` (non-root) afterwards to keep ownership clean.

**`ping 192.168.123.164` times out.**
- `ip link show <iface>` says `NO-CARRIER` → cable / port issue.
- Static IP not assigned → re-do §5.
- G1 not in Developer Mode.

**Pico controllers not detected.**
- `XRoboToolkit-PC-Service` not running on the PC.
- Headset and PC on different networks (corporate WiFi often blocks peer-to-peer).
- Run `python decoupled_wbc/control/teleop/streamers/pico_streamer.py` for a stand-alone debug stream.

**Robot triggers safety shutdown immediately at teleop start.**
Velocity limits in `decoupled_wbc/control/envs/g1/utils/joint_safety.py` too tight — see §3.

**Camera server: `No RealSense device found`.**
USB cable / port on the G1. Re-seat USB-C, then re-run `./start_camera_server.sh`.

**Image pull fails for `nvgear/gr00t_wbc:latest`.**
Try `docker login` if you've previously rate-limited. The image is public, so a fresh `docker pull nvgear/gr00t_wbc:latest` should succeed.

**LFS files are pointer text instead of binaries.**
`git lfs install && git lfs pull` from the repo root.

---

## 12. Reference

- Decoupled WBC docs: <https://nvlabs.github.io/GR00T-WholeBodyControl/references/decoupled_wbc.html>
- VR teleop guide: <https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html>
- ZMQ Manager / PICO VR tutorial: <https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html>
- XR Robotics PC service: <https://github.com/XR-Robotics/XRoboToolkit-PC-Service>
- Unitree G1 SDK guide: <https://support.unitree.com/home/en/G1_developer>
- LeRobot G1 WiFi guide: <https://huggingface.co/docs/lerobot/en/unitree_g1>
- Collected ETHRC dataset (Drive): <https://drive.google.com/drive/folders/1_oW6BgZoD8DEIGUjs_S4r_I4R6l4vZes?usp=sharing>

---

## 13. Status on this Workstation (snapshot)

Captured at the time of writing:

| Check                              | Status                                   |
|------------------------------------|------------------------------------------|
| Docker                             | installed (`29.4.3`)                     |
| NVIDIA driver / GPU                | OK (GTX 1650 visible to `nvidia-smi`)    |
| NVIDIA Container Toolkit           | **missing** — install per §1 before first run |
| Git LFS                            | installed (3.4.1)                        |
| Ethernet static IP `192.168.123.222/24` | present on `enp4s0`                  |
| Ethernet link to G1                | `NO-CARRIER` (cable not plugged in)      |
| Decoupled WBC image                | **not yet pulled** — run §4              |

Run §1 and §4 once (requires `sudo`), then you're ready for the every-session flow in §10.
