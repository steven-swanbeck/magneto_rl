#!/usr/bin/env python3
# %%
import numpy as np
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# %%
def train (env, path, rel_path, timesteps):
    # . Training    
    checkpoint_callback = CheckpointCallback(
        # save_freq=10,
        save_freq=100000,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # - Start from scratch or load specified weights
    model = DQN("MultiInputPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/", exploration_fraction=1.0, exploration_initial_eps=1.0, exploration_final_eps=0.2)
    # model = DQN.load(path + rel_path + 'breakpoint.zip', env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    print(model.policy)
    
    # # - Training
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)
        
        for i in range(10):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                env.render()
                counter += 1
        env.close()
        
    finally:
        # pass
        model.save(path + rel_path + 'breakpoint.zip')

def main ():
    path = '/home/steven/magneto_ws/src/magneto_rl/'
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
    rel_path = 'weights/'
    
    # . Training
    train (env, path, rel_path, 5000000)

# %%
if __name__ == "__main__":
    main()

# %%
