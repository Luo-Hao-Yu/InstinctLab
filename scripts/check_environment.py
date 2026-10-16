"""Check package imports, task registration and a short GPU physics simulation."""

import json
import os
import sys
import traceback
from importlib.metadata import version

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    import gymnasium as gym
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from instinct_rl.runners import OnPolicyRunner  # noqa: F401

    import instinctlab.tasks  # noqa: F401
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: F401

    assert torch.cuda.is_available(), "CUDA is unavailable"
    task = "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0"
    assert task in gym.registry, f"Task was not registered: {task}"

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0"))
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/EnvironmentCheckCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )
    sim.reset()
    cube.update(sim.get_physics_dt())
    initial_height = cube.data.root_pos_w[0, 2].item()
    for _ in range(10):
        cube.write_data_to_sim()
        sim.step(render=False)
        cube.update(sim.get_physics_dt())
    final_height = cube.data.root_pos_w[0, 2].item()
    assert final_height < initial_height, "Rigid body did not fall under gravity"
    print(
        "ENVIRONMENT_CHECK_PASSED "
        + json.dumps(
            {
                "packages": {
                    name: version(name)
                    for name in ("isaaclab", "instinct_rl", "instinctlab", "torch", "numpy")
                },
                "gpu": torch.cuda.get_device_name(0),
                "registered_instinct_tasks": len([name for name in gym.registry if name.startswith("Instinct-")]),
                "physics_steps": 10,
                "cube_initial_height": initial_height,
                "cube_final_height": final_height,
            }
        ),
        flush=True,
    )
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    # Kit's fast shutdown exits with status 0 and suppresses pending exceptions.
    os._exit(1)

# This short check writes no simulation artifacts. Kit's teardown can wait on
# background services; process exit releases its temporary GPU resources.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
