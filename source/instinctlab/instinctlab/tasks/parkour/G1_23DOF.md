# G1 23dof Parkour

The current shared terrain configuration specializes in a straight course of ten
90×60 cm rough platforms separated by 10–35 cm gaps, with three terrain types.
See [the gap-crossing task guide](GAP_CROSSING.md) for changes, validation, and training commands.

This task uses Unitree's official `g1_23dof.urdf`, its pelvis root, joint axes,
inertias and joint/motor limits. Source attribution and the pinned revision are in
`assets/resources/unitree_g1/g1_23dof.source.json`.
The model includes 12 leg joints, waist yaw, and five joints per arm including wrist roll.
Waist roll/pitch and both wrists' pitch/yaw are absent. The original 29dof task,
dataset and checkpoints remain separate.

## Prepare motion reference

The converted reference is already available locally at
`datasets/parkour/g1_23dof_motion_reference/`.
On a new checkout, first obtain the original Parkour dataset, then run from the project root:

```bash
./scripts/env.sh scripts/convert_parkour_g1_23dof.py
```

The converter selects 23 joints by name, converts the original torso-root motion
to pelvis-root poses with the original URDF's forward kinematics, and reverses
the waist-yaw sign to match the official pelvis-root joint convention. It keeps
the original world-space foot trajectories; `CONVERSION.json` records the errors,
source/output checksums, and conversion rules. The loader recomputes link poses
and velocities with the 23dof model. This is a projection of the existing motion,
not a new human-motion retargeting optimization. Joint references are clipped
to the official robot limits when loaded.

The converter refuses to replace existing files unless `--overwrite` is supplied.
Do not supply the original 29dof dataset directory directly to the 23dof task:
its root frame and waist-yaw convention differ.

## Train

```bash
./scripts/env.sh scripts/instinct_rl/train.py \
  --task=Instinct-Parkour-Target-Amp-G1-23dof-v0 \
  --headless --livestream=0 --num_envs=512
```

The defaults are 512 environments, 30,000 iterations and a checkpoint every 5,000 iterations.
Runs are saved to `logs/instinct_rl/g1_23dof_parkour/<run>/`.
Start a new run: a 29dof checkpoint has different policy/AMP dimensions and cannot
be resumed directly. The depth encoder architecture is unchanged, but no weights
are transferred automatically.

## Play with WebRTC

Replace `<run>` with a directory name from `logs/instinct_rl/g1_23dof_parkour`:

```bash
./scripts/env.sh source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-23dof-Play-v0 \
  --load_run=<run> --checkpoint=model_30000.pt \
  --num_envs=1 --livestream=2
```

Only one simulator can use the default streaming ports at a time. For playback
through the same local-network WebRTC setup, connect to the server's IP address.

## Export without playback

```bash
./scripts/env.sh source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-23dof-Play-v0 \
  --load_run=<run> --checkpoint=model_30000.pt \
  --num_envs=1 --headless --livestream=0 --exportonnx --export_only
```

The run's `exported/` directory contains:

| File | Input | Output |
| --- | --- | --- |
| `0-depth_encoder.onnx` | `[1, 8, 18, 32]` | `[1, 128]` |
| `actor.onnx` | `[1, 752]` | `[1, 23]` |
| `policy_metadata.json` | Observation layout, joint names and motor mapping | Action scales/offsets, PD gains, limits and policy timestep |

The policy receives 5,232 values: 624 proprioceptive/history values and 4,608
depth-history values. After depth encoding, the actor receives 624 + 128 values.
AMP policy and reference observations are both 550-dimensional.

**ONNX output follows the recorded PhysX joint order, not consecutive SDK indices.**
Use `action_joint_names` and `action_sdk_motor_indices` in the metadata together.
The SDK slots are sparse because the six absent motors retain their unused slots.
Convert normalized actions with the recorded scale and offset before using
position targets. The observation root is `pelvis`; a torso IMU must be converted
to that frame. Depth input is the configured cropped, normalized history, not a
raw depth image. These files describe the simulation interface; the physical
robot controller, camera extrinsics, delays and motor configuration still need
to match it. No physical robot commands are sent by these scripts.

## Verify a checkpoint and export

Offline numerical check without launching the simulator:

```bash
./scripts/env.sh scripts/check_parkour_onnx.py \
  --run-dir=logs/instinct_rl/g1_23dof_parkour/<run> \
  --checkpoint=model_30000.pt
```

Simulation check of joints, mirror FK, 29dof configuration regression, AMP dimensions,
and 32 steps driven by ONNX, compared against PyTorch actions:

```bash
./scripts/env.sh scripts/check_g1_23dof_parkour.py \
  --checkpoint=/workspace/instinctlab/logs/instinct_rl/g1_23dof_parkour/<run>/model_30000.pt \
  --headless --livestream=0
```

The 23dof task uses the official rubber hands and feet, without the custom shoes
of the original 29dof Parkour task. It reuses the Parkour PD gain and delay design;
motor limits come from the official 23dof URDF. These nominal values do not replace
checking the model revision and calibration of the physical robot.

## Local verification (2026-10-16)

- One training iteration with 8 environments and another with the default 512 environments completed and saved checkpoints.
- The Play task exported ONNX and metadata successfully with `--export_only`.
- Both ONNX graphs passed structural checks and 20 synthetic-input comparisons
  with the original PyTorch checkpoint (maximum action error `2.54e-7`).
- 32 ONNX-driven simulation steps passed, including comparison against PyTorch
  on actual observations (maximum action error `3.73e-8`).
- Mirror FK, SDK mapping, AMP dimensions and the existing 29dof configuration passed checks.

Summary: `.environment/g1-23dof-validation.json`; detailed logs:
`.environment/g1-23dof-training-check.log`, `.environment/g1-23dof-default-check.log`,
`.environment/g1-23dof-export-check.log`, `.environment/g1-23dof-play-check.log`,
and `.environment/g1-23dof-onnx-check.log`.
These checks validate the workflow, not learned Parkour performance. The generated
`model_1.pt` is a diagnostic checkpoint; a full 23dof training run and physical
robot validation have not been performed.
