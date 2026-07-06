# robot_are_learning

# Create and activate a clean environment
python3 -m venv mujoco_env
source mujoco_env/bin/activate

# Install Gymnasium (with MuJoCo support) and Stable-Baselines3
pip install "gymnasium[mujoco]" stable-baselines3
pip install tensorboard
