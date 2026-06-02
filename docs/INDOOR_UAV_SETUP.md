# Indoor-UAV physics-RL setup (habitat env)

Live RL needs habitat_sim + torch + gymnasium + gsplat in ONE process. We put
them all in the conda `habitat` env (py3.9):

  conda activate habitat
  export CUDA_HOME=/usr
  export PATH=$HOME/miniconda3/envs/habitat/bin:$PATH   # puts ninja on PATH for gsplat JIT

Installed: habitat-sim 0.3.3 (withbullet), torch 2.5.1+cu121, gymnasium 1.1.1,
gsplat 1.5.3 (CUDA backend JIT-compiles on first use; cached after).

Train (GS-reconstruction reward):
  python -m indoor_uav.scripts.train_explorer \
    --manifest indoor_uav/scenes_balanced_train.json --reward_mode gs \
    --run_dir logs/indoor_uav/<name> --total_steps 200000

Pieces: sim/drone_body.py (Bullet velocity-control drone, zero-grav hover,
real collisions) · tasks/coverage_map.py (nav/visited/frontier memory) ·
tasks/physics_coverage_env.py (RGB+map+state obs; geometric|gs reward) ·
policy/explorer_net.py (dual-CNN actor-critic) · scripts/train_explorer.py (PPO).
