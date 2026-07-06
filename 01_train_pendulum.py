import gymnasium as gym
from stable_baselines3 import PPO

def main():
    print("Initializing MuJoCo environment...")
    # Create the environment
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    print("Configuring PPO agent on CPU...")
    # Added device="cpu" to maximize efficiency for standard MuJoCo
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_tensorboard/",
        device="cpu" 
    )

    print("Starting training... Watch the window to see the robot learn!")
    try:
        model.learn(total_timesteps=20000)
        model.save("ppo_inverted_pendulum")
        print("Training complete! Model saved.")
    finally:
        # Putting this in a finally block ensures the GUI window closes gracefully 
        # even if you stop the script early with Ctrl+C
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()
