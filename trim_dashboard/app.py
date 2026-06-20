#!/usr/bin/env python3
"""
Episode Trim Dashboard for LeRobot datasets.

Creates a working copy of the dataset and serves a web UI that lets you
scrub through each episode and trim it at any frame. Parquet files, videos,
and all metadata are updated atomically on each trim.

Usage:
    conda activate lerobot-trim
    python trim_dashboard/app.py [--source <dir>] [--output <dir>] [--port 5000]
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, abort, jsonify, render_template, request, send_file

# ── Defaults ───────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

DEFAULT_SOURCE = os.path.join(_PROJECT, "outputs", "2026-05-27-12-07-25-G1-sim")
DEFAULT_OUTPUT = os.path.join(_PROJECT, "outputs", "2026-05-27-12-07-25-G1-sim-trimmed")
FFMPEG = "/home/rguntz/anaconda3/envs/lerobot-trim/bin/ffmpeg"

FPS = 20
VIDEO_KEYS = [
    "observation.images.ego_view",
    "observation.images.ego_view_left_mono",
    "observation.images.ego_view_right_mono",
]

# ── Training-key constants ─────────────────────────────────────────────────────

LOCOMANIP_KEY = "action.locomanip"
# action.eef(14) + teleop.navigate_command(3) + teleop.base_height_command(1)
LOCOMANIP_FEATURE = {
    "dtype": "float64",
    "shape": [18],
    "names": [
        "left_wrist_pos",
        "left_wrist_abs_quat",
        "right_wrist_pos",
        "right_wrist_abs_quat",
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "base_height_command",
    ],
}
LOCOMANIP_MODALITY = {
    "locomanip_left_wrist_pos":  {"start": 0,  "end": 3,  "original_key": LOCOMANIP_KEY},
    "locomanip_left_wrist_quat": {"start": 3,  "end": 7,  "original_key": LOCOMANIP_KEY, "rotation_type": "quaternion"},
    "locomanip_right_wrist_pos": {"start": 7,  "end": 10, "original_key": LOCOMANIP_KEY},
    "locomanip_right_wrist_quat":{"start": 10, "end": 14, "original_key": LOCOMANIP_KEY, "rotation_type": "quaternion"},
    "locomanip_navigate":        {"start": 14, "end": 17, "original_key": LOCOMANIP_KEY},
    "locomanip_base_height":     {"start": 17, "end": 18, "original_key": LOCOMANIP_KEY},
}

# ── Global state (set at startup) ──────────────────────────────────────────────

SOURCE_DIR: str = ""    # original raw dataset (no suffix)
DATASET_DIR: str = ""   # working copy for trimming (<source>-trimmed)
OUTPUTS_DIR: str = os.path.join(_PROJECT, "outputs")

app = Flask(__name__)

# ── Path helpers ───────────────────────────────────────────────────────────────


def _parquet_path(episode_index: int) -> str:
    return os.path.join(
        DATASET_DIR, "data", "chunk-000", f"episode_{episode_index:06d}.parquet"
    )


def _video_path(episode_index: int, video_key: str) -> str:
    return os.path.join(
        DATASET_DIR, "videos", "chunk-000", video_key, f"episode_{episode_index:06d}.mp4"
    )


# ── Metadata helpers ───────────────────────────────────────────────────────────


def _read_episodes() -> list[dict]:
    path = os.path.join(DATASET_DIR, "meta", "episodes.jsonl")
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return sorted(episodes, key=lambda e: e["episode_index"])


def _write_episodes(episodes: list[dict]) -> None:
    path = os.path.join(DATASET_DIR, "meta", "episodes.jsonl")
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def _read_stats() -> dict[int, dict]:
    path = os.path.join(DATASET_DIR, "meta", "episodes_stats.jsonl")
    stats: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                stats[entry["episode_index"]] = entry["stats"]
    return stats


def _write_stats(stats_dict: dict[int, dict]) -> None:
    path = os.path.join(DATASET_DIR, "meta", "episodes_stats.jsonl")
    with open(path, "w") as f:
        for ep_idx in sorted(stats_dict.keys()):
            entry = {"episode_index": ep_idx, "stats": stats_dict[ep_idx]}
            f.write(json.dumps(entry) + "\n")


def _update_info_total_frames(total_frames: int) -> None:
    path = os.path.join(DATASET_DIR, "meta", "info.json")
    with open(path) as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


# ── Stats computation ──────────────────────────────────────────────────────────


def _col_stats(table: pa.Table, col_name: str) -> dict:
    col = table.column(col_name)
    field = table.schema.field(col_name)
    ftype = field.type

    if pa.types.is_fixed_size_list(ftype) or pa.types.is_list(ftype) or pa.types.is_large_list(ftype):
        arr = np.stack([row.as_py() for row in col])  # (N, D)
        d = arr.shape[1] if arr.ndim > 1 else 1
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(len(arr))] * d,
        }
    else:
        vals = np.array(col.to_pylist(), dtype=float)
        return {
            "min": [float(vals.min())],
            "max": [float(vals.max())],
            "mean": [float(vals.mean())],
            "std": [float(vals.std())],
            "count": [int(len(vals))],
        }


def _recompute_episode_stats(episode_index: int) -> dict:
    table = pq.read_table(_parquet_path(episode_index))
    stats = {}
    for col_name in table.schema.names:
        try:
            stats[col_name] = _col_stats(table, col_name)
        except Exception:
            pass
    return stats


# ── Flask routes ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/episodes")
def api_episodes():
    return jsonify(_read_episodes())


@app.route("/api/video/<int:episode_index>")
def api_video(episode_index: int):
    # video_key can be passed as ?key=observation.images.ego_view
    key = request.args.get("key", VIDEO_KEYS[0])
    if key not in VIDEO_KEYS:
        abort(400)
    path = _video_path(episode_index, key)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/api/trim", methods=["POST"])
def api_trim():
    data = request.json
    episode_index = int(data["episode_index"])
    trim_frame = int(data["trim_frame"])  # last frame to KEEP (inclusive, 0-based)

    parquet_file = _parquet_path(episode_index)
    table = pq.read_table(parquet_file)
    original_length = len(table)
    new_length = trim_frame + 1  # number of rows to keep

    if new_length >= original_length:
        return jsonify({"error": "Trim point must be before the last frame"}), 400
    if new_length < 1:
        return jsonify({"error": "Must keep at least 1 frame"}), 400

    # 1. Trim parquet — slice preserves PyArrow schema + HuggingFace metadata
    trimmed = table.slice(0, new_length)
    pq.write_table(trimmed, parquet_file, compression="snappy")

    # 2. Trim all three video streams
    duration_s = new_length / FPS
    video_warnings: list[str] = []
    for vk in VIDEO_KEYS:
        vpath = _video_path(episode_index, vk)
        if not os.path.exists(vpath):
            continue
        tmp = vpath + ".tmp.mp4"
        cmd = [
            FFMPEG, "-y",
            "-i", vpath,
            "-t", str(duration_s),
            "-c", "copy",
            "-movflags", "+faststart",
            tmp,
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.replace(tmp, vpath)
        else:
            video_warnings.append(f"{vk}: {result.stderr[:300]}")
            if os.path.exists(tmp):
                os.remove(tmp)

    # 3. Update episodes.jsonl
    episodes = _read_episodes()
    for ep in episodes:
        if ep["episode_index"] == episode_index:
            ep["length"] = new_length
            break
    _write_episodes(episodes)

    # 4. Recompute stats for this episode
    stats_dict = _read_stats()
    stats_dict[episode_index] = _recompute_episode_stats(episode_index)
    _write_stats(stats_dict)

    # 5. Update info.json
    total = sum(ep["length"] for ep in episodes)
    _update_info_total_frames(total)

    resp = {
        "success": True,
        "episode_index": episode_index,
        "original_length": original_length,
        "new_length": new_length,
        "frames_removed": original_length - new_length,
    }
    if video_warnings:
        resp["warnings"] = video_warnings
    return jsonify(resp)


@app.route("/api/finalize", methods=["POST"])
def api_finalize():
    """
    Rewrite the global 'index' column across all episodes so indices are
    contiguous from 0. Call this once after all trimming is done.
    """
    episodes = _read_episodes()
    stats_dict = _read_stats()
    global_idx = 0

    for ep in episodes:
        ep_idx = ep["episode_index"]
        path = _parquet_path(ep_idx)
        table = pq.read_table(path)
        n = len(table)

        new_indices = pa.array(range(global_idx, global_idx + n), type=pa.int64())
        col_pos = table.schema.get_field_index("index")
        table = table.set_column(col_pos, "index", new_indices)
        pq.write_table(table, path, compression="snappy")

        # Update index stats
        if ep_idx in stats_dict:
            std = float(np.std(np.arange(n))) if n > 1 else 0.0
            stats_dict[ep_idx]["index"] = {
                "min": [global_idx],
                "max": [global_idx + n - 1],
                "mean": [global_idx + (n - 1) / 2.0],
                "std": [std],
                "count": [n],
            }

        global_idx += n

    _write_stats(stats_dict)
    _update_info_total_frames(global_idx)
    return jsonify({"success": True, "total_frames": global_idx})


# ── Dataset management ────────────────────────────────────────────────────────


@app.route("/api/datasets")
def api_datasets():
    """List raw datasets in outputs/ (excludes -trimmed and -augmented copies)."""
    if not os.path.exists(OUTPUTS_DIR):
        return jsonify([])
    entries = []
    for name in sorted(os.listdir(OUTPUTS_DIR)):
        if name.endswith("-trimmed") or name.endswith("-augmented"):
            continue
        full_path = os.path.join(OUTPUTS_DIR, name)
        if not os.path.isdir(full_path):
            continue
        if not os.path.exists(os.path.join(full_path, "meta", "info.json")):
            continue
        entries.append({
            "name": name,
            "active": full_path == SOURCE_DIR,
            "has_trimmed": os.path.exists(full_path + "-trimmed"),
            "has_augmented": os.path.exists(full_path + "-augmented"),
        })
    return jsonify(entries)


@app.route("/api/select_dataset", methods=["POST"])
def api_select_dataset():
    """Switch the active source dataset; creates the -trimmed working copy if missing."""
    global SOURCE_DIR, DATASET_DIR
    data = request.json
    source_name = data.get("source_name", "")
    source_path = os.path.join(OUTPUTS_DIR, source_name)

    if not os.path.isdir(source_path):
        return jsonify({"error": f"Dataset not found: {source_name}"}), 400
    if not os.path.exists(os.path.join(source_path, "meta", "info.json")):
        return jsonify({"error": f"Not a valid LeRobot dataset: {source_name}"}), 400

    trimmed_path = source_path + "-trimmed"
    created = not os.path.exists(trimmed_path)
    if created:
        shutil.copytree(source_path, trimmed_path)

    SOURCE_DIR = source_path
    DATASET_DIR = trimmed_path

    return jsonify({"success": True, "source_name": source_name, "created_trimmed": created})


@app.route("/api/state")
def api_state():
    """Return the current active source info for UI initialisation."""
    source_name = os.path.basename(SOURCE_DIR) if SOURCE_DIR else None
    return jsonify({
        "source_name": source_name,
        "has_trimmed":   bool(SOURCE_DIR) and os.path.exists(SOURCE_DIR + "-trimmed"),
        "has_augmented": bool(SOURCE_DIR) and os.path.exists(SOURCE_DIR + "-augmented"),
    })


# ── Training keys ─────────────────────────────────────────────────────────────


def _build_locomanip_column(table: pa.Table) -> pa.Array:
    eef    = np.array(table.column("action.eef").to_pylist(), dtype=np.float64)               # (N, 14)
    nav    = np.array(table.column("teleop.navigate_command").to_pylist(), dtype=np.float64)  # (N, 3)
    height = np.array(table.column("teleop.base_height_command").to_pylist(), dtype=np.float64).reshape(-1, 1)  # (N, 1)
    combined = np.concatenate([eef, nav, height], axis=1)                                     # (N, 18)
    return pa.array(combined.tolist())


def _do_create_training_keys(target_dir: str) -> dict:
    """Add action.locomanip to every parquet in target_dir and update its meta files."""
    episodes = []
    with open(os.path.join(target_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes.sort(key=lambda e: e["episode_index"])

    if not episodes:
        return {"error": "No episodes found in target dataset."}

    first_path = os.path.join(
        target_dir, "data", "chunk-000",
        f"episode_{episodes[0]['episode_index']:06d}.parquet",
    )
    if LOCOMANIP_KEY in pq.read_table(first_path).schema.names:
        return {"error": f"'{LOCOMANIP_KEY}' already exists in this dataset."}

    stats_dict: dict[int, dict] = {}
    with open(os.path.join(target_dir, "meta", "episodes_stats.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                stats_dict[entry["episode_index"]] = entry["stats"]

    errors = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        path = os.path.join(target_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet")
        try:
            table = pq.read_table(path)
            table = table.append_column(LOCOMANIP_KEY, _build_locomanip_column(table))
            pq.write_table(table, path, compression="snappy")
            if ep_idx in stats_dict:
                stats_dict[ep_idx][LOCOMANIP_KEY] = _col_stats(table, LOCOMANIP_KEY)
        except Exception as exc:
            errors.append(f"Episode {ep_idx}: {exc}")

    with open(os.path.join(target_dir, "meta", "episodes_stats.jsonl"), "w") as f:
        for ep_idx in sorted(stats_dict.keys()):
            f.write(json.dumps({"episode_index": ep_idx, "stats": stats_dict[ep_idx]}) + "\n")

    info_path = os.path.join(target_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    info["features"][LOCOMANIP_KEY] = LOCOMANIP_FEATURE
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    modality_path = os.path.join(target_dir, "meta", "modality.json")
    with open(modality_path) as f:
        modality = json.load(f)
    modality["action"].update(LOCOMANIP_MODALITY)
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=4)

    return {"processed": len(episodes) - len(errors), "errors": errors}


@app.route("/api/create_training_keys", methods=["POST"])
def api_create_training_keys():
    if not SOURCE_DIR or not DATASET_DIR:
        return jsonify({"error": "No dataset selected. Pick a dataset first."}), 400

    augmented_dir = SOURCE_DIR + "-augmented"
    augmented_name = os.path.basename(augmented_dir)

    if os.path.exists(augmented_dir):
        return jsonify({"error": f"Augmented dataset already exists: {augmented_name}"}), 400

    shutil.copytree(DATASET_DIR, augmented_dir)

    result = _do_create_training_keys(augmented_dir)
    if "error" in result:
        shutil.rmtree(augmented_dir)
        return jsonify(result), 400

    resp = {"success": True, "processed": result["processed"], "augmented_name": augmented_name}
    if result["errors"]:
        resp["errors"] = result["errors"]
    return jsonify(resp)


# ── Startup ────────────────────────────────────────────────────────────────────


def setup(source_dir: str, output_dir: str) -> None:
    global SOURCE_DIR, DATASET_DIR
    SOURCE_DIR = source_dir
    DATASET_DIR = output_dir

    if not os.path.exists(output_dir):
        print(f"\nSource : {source_dir}")
        print(f"Output : {output_dir}")
        print("\nCreating working copy — this may take ~30-60 s for a 1.2 GB dataset...")
        shutil.copytree(source_dir, output_dir)
        print("Copy complete.\n")
    else:
        print(f"\nUsing existing working copy:\n  {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode Trim Dashboard")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    setup(args.source, args.output)

    print(f"Dashboard → http://localhost:{args.port}")
    print("Keyboard shortcuts: ← → (frame step)  Space (play/pause)  T (trim)\n")
    app.run(debug=False, host="0.0.0.0", port=args.port)
