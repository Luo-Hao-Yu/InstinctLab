# Parkour Task

For the standard Unitree G1 23dof robot, use the separate
[G1 23dof training, playback and export workflow](G1_23DOF.md).
The task IDs below retain the original 29dof model.

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=<run_name> --num_envs=1 --exportonnx --useonnx
```

To export and exit without playback or livestreaming:

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=<run_directory> --checkpoint=model_30000.pt --num_envs=1 --headless --livestream=0 --exportonnx --export_only
```

`--load_run` selects the run directory; `--checkpoint` selects the model file inside it.
Export still initializes one simulation environment to obtain the policy input, then writes
the ONNX files to `<run_directory>/exported/` and exits before the playback loop.

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--export_only`: Exit after export without playback (requires `--exportonnx` and `--load_run`)
- `--useonnx`: Use existing exported ONNX models for inference during playing, or combine with `--exportonnx` to export first
