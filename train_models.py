from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from maze_game import MazeGame
from models import MLP, RNN, LNN

if __name__ == "__main__":
    env = make_vec_env(lambda: MazeGame(visualize=False), n_envs=4)

    mlp_model = PPO(policy=MLP, env=env, verbose=1)
    mlp_model.learn(total_timesteps=10000, progress_bar=True)
    
    rnn_model = PPO(policy=RNN, env=env, verbose=1)
    rnn_model.learn(total_timesteps=10000, progress_bar=True)

    lnn_model = PPO(policy=LNN, env=env, verbose=1)
    lnn_model.learn(total_timesteps=1000000, progress_bar=True)
    lnn_model.save("trained_lnn")