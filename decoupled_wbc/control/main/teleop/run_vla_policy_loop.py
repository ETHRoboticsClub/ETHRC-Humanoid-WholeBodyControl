"""VLA control loop: swap the teleop policy for a remote GR00T VLA.

This is the structural analogue of ``run_teleop_policy_loop.py`` except that
the goal stream comes from an Isaac-GR00T PolicyServer rather than Pico/Leap
teleop devices. It:

  1. Subscribes to the robot's proprio stream (STATE_TOPIC_NAME) for joint ``q``.
  2. Connects to the camera server for the ``ego_view`` image.
  3. Queries the VLA server for action chunks and unrolls them step-by-step.
  4. Publishes the per-step goal to ``CONTROL_GOAL_TOPIC``, which the main
     control loop (``run_g1_control_loop.py``) turns into ``G1DecoupledWholeBodyPolicy.set_goal``.

Run ``run_g1_control_loop.py`` in one terminal and this script in another.
"""
from dataclasses import dataclass
import time

import rclpy
import tyro

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_MODEL_SERVER_PORT,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.policy.gr00t_client_policy import Gr00tClientPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardDispatcher
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
)
from decoupled_wbc.control.utils.telemetry import Telemetry

VLA_NODE_NAME = "VLAPolicy"


@dataclass
class VLAPolicyConfig:
    """CLI config for the VLA policy loop."""

    task_prompt: str
    """Natural-language instruction sent to the VLA on every inference call."""

    server_host: str = "localhost"
    """Host running Isaac-GR00T's ``run_gr00t_server.py``."""

    server_port: int = DEFAULT_MODEL_SERVER_PORT
    """Port the VLA server is listening on."""

    camera_host: str = "localhost"
    """Host running the decoupled_wbc camera server."""

    camera_port: int = 5555
    """Camera server port (ComposedCameraServer default)."""

    control_frequency: float = 20.0
    """Control tick rate. Must be <= the VLA action horizon / server round-trip."""

    refresh_every: int = 15
    """Force a new VLA query after this many dispatched steps (chunk len ~= 30)."""

    enable_waist: bool = False
    """Mirror the flag used by ``run_g1_control_loop.py`` so joint groups line up."""

    high_elbow_pose: bool = False
    """Mirror the flag used by ``run_g1_control_loop.py``."""

    time_to_initial_pose: float = 2.0
    """Seconds to extend ``target_time`` on the first ACTIVE tick of an
    episode so the IK interpolator can settle out of the home pose."""

    start_key: str = "s"
    """Keyboard key (in this terminal) that transitions IDLE → ACTIVE."""

    terminate_key: str = "t"
    """Keyboard key (in this terminal) that transitions ACTIVE → IDLE
    and drives the robot back to the home pose."""


def main(config: VLAPolicyConfig):
    ros_manager = ROSManager(node_name=VLA_NODE_NAME)
    node = ros_manager.node

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=config.high_elbow_pose,
    )

    # Proprio subscription — feeds VLA state modality.
    state_sub = ROSMsgSubscriber(STATE_TOPIC_NAME)

    # Image source — feeds VLA video modality.
    camera = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    policy = Gr00tClientPolicy(
        robot_model=robot_model,
        camera=camera,
        task_prompt=config.task_prompt,
        server_host=config.server_host,
        server_port=config.server_port,
        refresh_every=config.refresh_every,
        dt=1.0 / config.control_frequency,
        camera_endpoint=f"{config.camera_host}:{config.camera_port}",
        time_to_initial_pose=config.time_to_initial_pose,
        start_key=config.start_key,
        terminate_key=config.terminate_key,
    )

    # Health check the server before entering the hot loop.
    if not policy.client.ping():
        raise RuntimeError(
            f"VLA server at {config.server_host}:{config.server_port} is not responding."
        )
    print(
        f"[vla_loop] connected to VLA server {config.server_host}:{config.server_port}, "
        f"task='{config.task_prompt}'"
    )
    print(
        f"[vla_loop] diagnostics: proprio topic '{STATE_TOPIC_NAME}', "
        f"camera ZMQ tcp://{config.camera_host}:{config.camera_port} "
        f"(see [gr00t_client] lines if something blocks)."
    )
    print(
        f"[vla_loop] IDLE — waiting for episode start. "
        f"press '{config.start_key}' to start an episode, "
        f"'{config.terminate_key}' to terminate."
    )

    keyboard_dispatcher = KeyboardDispatcher()
    keyboard_dispatcher.register(policy)
    keyboard_dispatcher.start()

    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    rate = node.create_rate(config.control_frequency)
    telemetry = Telemetry(window_size=100)
    logged_state_missing_q = False

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()

                # Pull latest proprio into the policy.
                msg = state_sub.get_msg()
                if msg is not None:
                    if "q" not in msg:
                        if not logged_state_missing_q:
                            print(
                                f"[vla_loop] state message on '{STATE_TOPIC_NAME}' has no "
                                "'q' key; cannot feed VLA until payload matches publisher."
                            )
                            logged_state_missing_q = True
                    else:
                        policy.set_observation({"q": msg["q"]})

                # Compute + publish the next goal.
                with telemetry.timer("get_action"):
                    goal = policy.get_action()

                with telemetry.timer("publish_goal"):
                    control_publisher.publish(goal)

                end_time = time.monotonic()
                if (end_time - t_start) > (1.0 / config.control_frequency):
                    telemetry.log_timing_info(
                        context="VLA Policy Loop Missed", threshold=0.001
                    )
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"[vla_loop] interrupted: {e}")

    finally:
        print("[vla_loop] cleaning up...")
        keyboard_dispatcher.stop()
        ros_manager.shutdown()


if __name__ == "__main__":
    main(tyro.cli(VLAPolicyConfig))
