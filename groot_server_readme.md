 Files created

  - GR00T-WholeBodyControl/decoupled_wbc/control/policy/gr00t_client_policy.py — Gr00tClientPolicy. Wraps an Isaac-GR00T PolicyClient, builds the VLA observation
   (video / state-slices-from-q / language), unrolls the 30-step action chunk into per-tick goals, and maps left_arm|right_arm|left_hand|right_hand|waist back
  into target_upper_body_pose using robot_model.get_joint_group_indices.
  - GR00T-WholeBodyControl/decoupled_wbc/control/main/teleop/run_vla_policy_loop.py — the control-loop wrapper, structurally a 1:1 swap for
  run_teleop_policy_loop.py. Subscribes to STATE_TOPIC_NAME, connects to the camera, and publishes goals to CONTROL_GOAL_TOPIC.

  How to run

  Three processes, in this order:

  # 1. VLA inference server (Jetson or dGPU)
  cd Isaac-GR00T
  source scripts/activate_orin.sh   # or activate_thor.sh / activate_spark.sh
  python -m gr00t.eval.run_gr00t_server \
      --model-path /path/to/your/finetuned_ckpt \
      --embodiment-tag UNITREE_G1 \
      --host 0.0.0.0 --port 5555

  # 2. G1 control loop (decoupled_wbc docker, robot host)
  python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real
  #  (or --interface sim for MuJoCo testing)

  # 3. VLA client loop (same machine as control loop)
  python decoupled_wbc/control/main/teleop/run_vla_policy_loop.py \
      --task-prompt "pick up the bottle and place it on the plate" \
      --server-host <jetson-ip> --server-port 5555 \
      --camera-host localhost --camera-port 5555 \
      --control-frequency 20 --refresh-every 15

  If you pass --enable-waist/--high-elbow-pose to the control loop, pass the same to the VLA client so joint-group indices match.

  Design notes / caveats

  - Action decoding is handled server-side. Gr00tPolicy._get_action calls processor.decode_action(...) which unnormalizes and adds state for RELATIVE
  representations. The returned arrays are absolute joint targets, not deltas — safe to drop into target_upper_body_pose.
  - Upper-body assembly is robust to waist_location. If waist isn't in upper_body in the current robot_model, the VLA's waist prediction is silently dropped (the
   lower body policy handles it). If it is, it's placed at its correct index.
  - Chunk cadence. Chunk length = 30 (from the unitree_g1 modality delta_indices=list(range(30))). Default refresh_every=15 gives a 0.75 s chunk stride at 20 Hz
  — keeps the VLA responsive to new observations while amortizing server latency. Set --refresh-every 30 for open-loop-per-chunk, or --refresh-every 1 for fully
  closed-loop (latency-bound).
  - Safety: while proprio hasn't arrived yet, get_action returns an initial_upper_body_pose + zero nav/default height, so the WBC interpolates to a known-safe
  state rather than NaNs.
  - Camera key. Hard-coded to ego_view (the key exporter writes). If we collected with add_stereo_camera=True, the model is only trained on ego_view anyway
   per the pre-registered modality, so this is fine.
  - Key-name quirk. The canonical goal-dict uses base_height_command + navigate_cmd (per pico_streamer.py and G1DecoupledWholeBodyPolicy.set_goal).
  LerobotReplayPolicy returns base_height_cmd which looks like a long-standing typo in that file — my client uses the canonical names.
  - Imports: this file imports from gr00t.policy.server_client, so Isaac-GR00T must be importable in the decoupled_wbc env. The cheapest way: pip install -e
  /path/to/Isaac-GR00T --no-deps in the decoupled_wbc container. You don't need the VLA model code pulled in at runtime — only PolicyClient (zmq + msgpack).

  Things I did not do 

  - Didn't verify action-key dimensionalities against a real meta/modality.json produced by our exporter. If our robot_model configuration yields e.g. a waist
  size different from what the checkpoint was trained with, upper_pose[positions] = vals[mask] will raise. Worth dumping one frame of observation + action shapes
   on the first _refresh_chunk call to confirm.
  - No hot-reload/reconnect on the ZMQ socket beyond what PolicyClient._init_socket already does — transient network hiccups will surface as RuntimeError from
  the control loop.
  - Didn't add toggle_policy_action / toggle_stand_command to the goal dict, so the lower-body policy stays in whatever mode run_g1_control_loop.py keyboard
  controls left it in (] to activate). If you want the VLA to be able to activate locomotion autonomously, add those keys to the emitted goal.