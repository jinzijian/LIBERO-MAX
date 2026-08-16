#!/usr/bin/env python3
"""Load VLA-JEPA and exercise the exact MAX action conversion without MuJoCo."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vlajepa_persistent_shard import _to_libero_actions, _unnormalize_actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlajepa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vlajepa_root))

    import torch
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.tools import read_mode_config

    model_config, norm_stats = read_mode_config(args.checkpoint)
    chunk_size = (
        int(model_config["framework"]["action_model"]["future_action_window_size"])
        + 1
    )
    model = baseframework.from_pretrained(str(args.checkpoint.absolute()))
    model = model.to(torch.bfloat16).to(torch.device("cuda:0")).eval()
    rng = np.random.default_rng(195)
    images = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
    ]
    with torch.no_grad():
        response = model.predict_action(
            batch_images=to_pil_preserve([images]),
            instructions=["pick up the alphabet soup and place it in the basket"],
            unnorm_key="franka",
            do_sample=False,
            use_ddim=True,
            num_ddim_steps=10,
            state=[np.zeros((1, 8), dtype=np.float32)],
        )
    normalized = np.asarray(response["normalized_actions"])[0]
    raw = _unnormalize_actions(normalized, norm_stats["franka"]["action"])
    actions = _to_libero_actions(raw)
    if actions.shape != (chunk_size, 7) or not np.isfinite(actions).all():
        raise RuntimeError("invalid converted actions: %s" % (actions.shape,))
    print("status=complete")
    print("query_interval=%d" % chunk_size)
    print("normalized_shape=%s" % (normalized.shape,))
    print("libero_shape=%s" % (actions.shape,))
    print("gripper_values=%s" % sorted(set(actions[:, 6].tolist())))
    print("gpu=%s" % torch.cuda.get_device_name(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
