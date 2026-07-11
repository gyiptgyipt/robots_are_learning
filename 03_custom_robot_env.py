import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3 import PPO

# 1. Define the physical robot via MuJoCo XML
# A base with a single hinge joint and a link extending outwards
ROBOT_XML = """
<mujoco model="simple_arm">
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <light pos="0 0 3"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.05 0.1" rgba="0.5 0.5 0.5 1"/>
            <body name="arm" pos="0 0 0.1">
                <joint name="joint1" type="hinge" axis="0 1 0" range="-180 180"/>
                <geom type="capsule" size="0.02 0.2" fromto="0 0 0 0 0 0.4" rgba="0 0 1 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1" gear="5.0"/>
    </actuator>
</mujoco>
"""

# 2. Build the Custom Gymnasium Wrapper
class CustomArmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Initialize MuJoCo model and data from the XML string
        self.model = mujoco.MjModel.from_xml_string(ROBOT_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = None

        # Action Space: Continuous torque command between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [sin(angle), cos(angle), angular_velocity]
        # Using sin/cos avoids the discontinuity wrap-around at 360 degrees
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Target angle we want the robot to reach: 1.57 radians (~90 degrees)
        self.target_angle = 1.57 

    def _get_obs(self):
        # Read the current joint position and velocity from MuJoCo
        angle = self.data.qpos[0]
        velocity = self.data.qvel[0]
        return np.array([np.sin(angle), np.cos(angle), velocity], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset physics state
        mujoco.mj_resetData(self.model, self.data)
        
        # Start the arm with a small random position so it learns to generalize
        self.data.qpos[0] = np.random.uniform(-0.5, 0.5)
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        # Apply the action torque to the actuator array
        self.data.ctrl[0] = action[0]

        # Step the physics engine forward by 1 timestep
        mujoco.mj_step(self.model, self.data)

        # Get updated observations
        obs = self._get_obs()
        current_angle = self.data.qpos[0]

        # --- THE REWARD FUNCTION ---
        # 1. Calculate the error distance to the target angle
        angle_error = np.abs(current_angle - self.target_angle)
        # 2. Give higher rewards for being closer to the target
        reward = -angle_error 
        # 3. Penalize excessive movement/shaking when close
        reward -= 0.01 * np.abs(self.data.qvel[0]) 

        # Termination criteria (keep it running for a fixed time limit)
        terminated = False
        truncated = False # Handled externally by time limits if wrapped, or left False here

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from mujoco.viewer import launch_passive
                self.renderer = launch_passive(self.model, self.data)
            self.renderer.sync()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

# 3. Execution Main Block
if __name__ == "__main__":
    # Create training env (without rendering to speed up calculation)
    train_env = CustomArmEnv()
    
    print("Training custom robot arm...")
    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
    model.learn(total_timesteps=40000)
    train_env.close()
    print("Training complete!")

    # Evaluate the trained model in real-time
    print("Running evaluation... Watch the arm try to hold 90 degrees!")
    eval_env = CustomArmEnv(render_mode="human")
    obs, info = eval_env.reset()
    
    try:
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                obs, info = eval_env.reset()
    finally:
        eval_env.close()