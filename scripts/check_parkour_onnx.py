"""Compare a Parkour ONNX export with its checkpoint using policy_metadata.json (no simulator)."""

import argparse
import hashlib
import importlib.util
import json
import numpy as np
import torch
import yaml
from pathlib import Path

import instinct_rl.modules as modules
import onnx
import onnxruntime as ort
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


def check(run_dir, checkpoint_name, samples):
    exported = run_dir / "exported"
    metadata = json.loads((exported / "policy_metadata.json").read_text())
    config = yaml.safe_load((run_dir / "params/agent.yaml").read_text())
    if config.get("empirical_normalization"):
        raise ValueError("This checker requires a policy without empirical observation normalization")
    policy_cfg = config["policy"]
    policy_class = getattr(modules, policy_cfg.pop("class_name"))
    obs_format = {
        group: {name: tuple(shape) for name, shape in segments.items()}
        for group, segments in metadata["observation_formats"].items()
    }
    policy = policy_class(obs_format, len(metadata["action_joint_names"]), **policy_cfg).eval()
    checkpoint_path = run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    models = {}
    for filename in ("0-depth_encoder.onnx", "actor.onnx"):
        path = exported / filename
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        models[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "opset": {item.domain or "ai.onnx": item.version for item in model.opset_import},
            "input": session.get_inputs()[0].shape,
            "output": session.get_outputs()[0].shape,
        }
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "parkour_onnxer", root / "source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py"
    )
    onnxer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(onnxer)
    segments = obs_format["policy"]
    proprio_size = get_subobs_size(segments, [name for name in segments if name != "depth_image"])
    onnx_policy = onnxer.load_parkour_onnx_model(
        str(exported),
        lambda obs: get_subobs_by_components(obs, ["depth_image"], segments, temporal=True),
        segments["depth_image"],
        slice(0, proprio_size),
    )
    max_error = 0.0
    with torch.inference_mode():
        for index in range(samples):
            scale = (0.0, 0.1, 1.0, 3.0)[index % 4]
            obs = torch.randn(1, int(get_subobs_size(segments))) * scale
            obs[:, proprio_size:] = torch.rand_like(obs[:, proprio_size:]) if scale else 0.0
            expected = policy.act_inference(obs).numpy()
            actual = onnx_policy(obs).numpy()
            assert actual.shape == (1, len(metadata["action_joint_names"]))
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    report = {
        "status": "PASSED",
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint["iter"],
        "samples": samples,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "max_absolute_error": max_error,
        "tolerance": {"atol": 1e-4, "rtol": 1e-4},
        "models": models,
    }
    (exported / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_30000.pt")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    check(args.run_dir, args.checkpoint, args.samples)
