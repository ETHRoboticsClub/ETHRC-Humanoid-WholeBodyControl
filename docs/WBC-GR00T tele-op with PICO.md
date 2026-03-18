# Groot WBC

<aside>
🤖

Guide for running the **Decoupled WBC** stack (`decoupled_wbc/`) on the **Unitree G1** — covering control, teleoperation, and data collection.

</aside>

- **Collected dataset**: [Google Drive](https://drive.google.com/drive/folders/1_oW6BgZoD8DEIGUjs_S4r_I4R6l4vZes?usp=sharing)

# System Installation

See: https://nvlabs.github.io/GR00T-WholeBodyControl/references/decoupled_wbc.html

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with a recent driver
- Docker and NVIDIA Container Toolkit (required for GPU access inside the container)

## Repository Setup

Install Git and Git LFS:

```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Isaac-GR00T.git
cd decoupled_wbc
```

## Docker Environment

We provide a Docker image with all dependencies pre-installed. **All commands must be run from inside the Docker container.**

Install a fresh image and start a container:

```bash
./docker/run_docker.sh --install --root
```

This pulls the latest `decoupled_wbc` image from `docker.io/nvgear`.

Start or re-enter a container:

```bash
./docker/run_docker.sh --root
```

Use `--root` to run as the `root` user. To run as a normal user, build the image locally:

```bash
./docker/run_docker.sh --build
```

## Network Setup (Real Robot)

Connect your workstation to the G1 via ethernet and configure a static IP on the same subnet as per the [G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer):

```bash
sudo ip addr add 192.168.123.222/24 dev enp6s0
sudo ip link set enp6s0 up
```

The G1's default IP is `192.168.123.164`. Your workstation must be `192.168.123.222` on the same `/24` subnet. Verify connectivity:

```bash
ping 192.168.123.164
```

For WiFi connectivity setup, see the [LeRobot Unitree G1 guide](https://huggingface.co/docs/lerobot/en/unitree_g1).

## Safety Limits

<aside>
🛡️

On a fresh clone you **must** update the velocity limits in `decoupled_wbc/control/envs/g1/utils/joint_safety.py`. If the robot exceeds these limits during teleoperation, the system triggers an immediate safety shutdown. The defaults are too conservative for teleoperation.

</aside>

| Parameter | Default | Recommended |
| --- | --- | --- |
| `ARM_VELOCITY_LIMIT` | 6.0 rad/s | 10.0 rad/s |
| `HAND_VELOCITY_LIMIT` | 50.0 rad/s | 70.0 rad/s |

---

# Running the Control Stack

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `]` | Activate policy |
| `o` | Deactivate policy |
| `9` | Release / Hold the robot |
| `w` / `s` | Move forward / backward |
| `a` / `d` | Strafe left / right |
| `q` / `e` | Rotate left / right |
| `z` | Zero navigation commands |
| `1` / `2` | Raise / lower base height |
| `backspace` | Reset robot in visualizer |

---

# Running the Teleoperation Stack

### Pico Setup and Controls

Configure the teleop app on your Pico headset by following the [XR Robotics guidelines](https://github.com/XR-Robotics). The necessary PC software is pre-installed in the Docker container — only the [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) component is needed.

**Prerequisite:** Connect the Pico to the same network as the host computer.

### Controller Bindings

| Input | Action |
| --- | --- |
| `menu + left trigger` | Toggle lower-body policy |
| `menu + right trigger` | Toggle upper-body policy |
| `Left stick` | X/Y translation |
| `Right stick` | Yaw rotation |
| `L/R triggers` | Control hand grippers |

Pico unit test:

```bash
python decoupled_wbc/control/teleop/streamers/pico_streamer.py
```

---

# Resources

- **Collected dataset**: [Google Drive](https://drive.google.com/drive/folders/1_oW6BgZoD8DEIGUjs_S4r_I4R6l4vZes?usp=sharing)
- **WiFi setup guide**: [LeRobot Unitree G1](https://huggingface.co/docs/lerobot/en/unitree_g1)

# Starting Tele-Op stack on G1

<aside>
🚀

Follow these steps **every time** you want to run the system on the real G1.

</aside>

1. **Power on the G1 and enter Developer Mode** — use the physical controller: press `L2+R2`, then `L2+A`, then `L2+B`. The robot should announce *"Developer mode"*.
2. **Connect ethernet** — plug the ethernet cable between the G1 and your workstation.
3. **Verify connectivity** — run `ping 192.168.123.164` from your workstation. If it fails, see the Network Setup section below.
4. **Set up the Pico headset** — put on the headset and open the teleop application. On the PC, also launch the XRoboToolkit-PC-Service companion app.
5. **Verify Pico connection** — make sure the headset and the PC are on the same local network (e.g. create a hotspot from your smartphone and connect both devices to it). Both should report that they are connected.
6. **Start the camera** — SSH into **the G1** and run:

```bash
./start_camera_server.sh
```

1. **Start the Docker container** — run `./decoupled_wbc/docker/run_docker.sh --root` in a terminal.
2. **Deploy the full stack** — open 1 terminal inside the Docker contrainer and run deploy G1 script. Otherwise, open 3 terminals inside the Docker container and run the 3 commands from the Run Commands section below (one per terminal).
3. The following commands might be helpful when running the policy on the G1 :
    
    Operations on Pico controllers:
    
    - `A`: Start/Stop recording
    - `B`: Discard trajectory
    
    Operations in the `controller` window (`control_data_teleop` pane, left):
    
    - `]`: Activate policy
    - `o`: Deactivate policy
    - `k`: Reset the simulation and policies
    - ```: Terminate the tmux session
    - `ctrl + d`: Exit the shell in the pane

---

# Deploy G1 script

```bash
python decoupled_wbc/scripts/deploy_g1.py \    
      --interface real \          
      --hand_control_device pico \      
      --body_control_device pico \
      --camera_host 192.168.123.164 \                        
      --no-add-stereo-camera
```

# Run Commands (be found in main of ETH RC WBC-GR00T)

<aside>
▶️

The three core processes — run each in a separate terminal inside the Docker container.

</aside>

**1. Control Loop**

```bash
# after this command, press ] to activate the policy
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real
```

**2. Tele-operation**

```bash
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --hand_control_device=pico \
    --body_control_device=pico
```

**3. Data Exporter**

```bash
python decoupled_wbc/control/main/teleop/run_g1_data_exporter.py \
    --camera-host 192.168.123.164 --no-add-stereo-camera
```