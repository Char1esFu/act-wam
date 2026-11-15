"""Convert raw UniVTAC episodes of one task into ACT training episodes.

    python process_data.py put_bottle_in_shelf clean 100 --raw-root /path/to/UniVTAC/data

writes data/sim-<task>-<config>-<num>/... and registers the dataset in SIM_TASK_CONFIGS.json,
which imitate_episodes.py reads (--task_name sim-<task>-<config>-<num>).
"""
import argparse
import json
import os
from pathlib import Path

from univtac_data import BaseDataPreprocessor

SIM_TASK_CONFIGS_PATH = Path(__file__).parent / "SIM_TASK_CONFIGS.json"
# Per-task camera setup (camera_type: head, or all = head + wrist), copied from UniVTAC policy/task_settings.json.
TASK_SETTINGS_PATH = Path(__file__).parent / "task_settings.json"


def task_cameras(task_name):
    with open(TASK_SETTINGS_PATH, "r") as f:
        camera_type = json.load(f).get(task_name, {}).get("camera_type", "head")
    return ["head", "wrist"] if camera_type == "all" else [camera_type]


def main(task_name, task_config, expert_data_num, raw_root, out_root, visual_cameras):
    output_path = Path(out_root) / f"sim-{task_name}" / f"{task_config}-{expert_data_num}"
    processor = BaseDataPreprocessor(task_name, task_config, raw_root)
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=["left", "right"],
        downsample_factor=1,
        episode_num=expert_data_num,
        random_select=False,
    )

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            sim_task_configs = json.load(f)
    except Exception:
        sim_task_configs = {}
    sim_task_configs[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": metadata["dataset_dir"],
        "num_episodes": metadata["num_episodes"],
        "episode_len": metadata["episode_len"],
        "camera_names": metadata["camera_names"],
    }
    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(sim_task_configs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw UniVTAC episodes into ACT episodes.")
    parser.add_argument("task_name", help="e.g. put_bottle_in_shelf")
    parser.add_argument("task_config", help="collection config, e.g. clean")
    parser.add_argument("expert_data_num", type=int, help="number of episodes to convert")
    parser.add_argument("--raw-root", default=os.environ.get("UNIVTAC_DATA", "raw_data"),
                        help="directory with <task>/<config>/<episode>.hdf5 (default: $UNIVTAC_DATA or raw_data)")
    parser.add_argument("--out-root", default="data", help="where the ACT episodes are written")
    parser.add_argument("--visual-cameras", nargs="+", default=None,
                        help="cameras to export (head, wrist); default: camera_type in task_settings.json")
    args = parser.parse_args()
    main(args.task_name, args.task_config, args.expert_data_num, args.raw_root, args.out_root,
         args.visual_cameras or task_cameras(args.task_name))
