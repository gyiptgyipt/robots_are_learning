import gymnasium as gym
from stable_baselines3 import PPO

def main():
    print("Loading MuJoCo environment...")
    # 1. Recreate the environment (using human render mode so we can watch it)
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    print("Loading saved PPO model...")
    # 2. Load the trained brain from the zip file
    # We specify device="cpu" just like we did during training
    model = PPO.load("ppo_inverted_pendulum", env=env, device="cpu")

    # 3. Evaluation Loop
    obs, info = env.reset()
    print("Running evaluation. Press Ctrl+C in the terminal to exit.")
    
    try:
        while True:
            # Ask the model to predict the next action based on the current observation
            # deterministic=True ensures the agent plays perfectly without exploring random actions
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the simulation forward using that action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # If the robot falls over or the episode finishes, reset it automatically
            if terminated or truncated:
                obs, info = env.reset()
                print("Environment reset!")
                
    finally:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()
