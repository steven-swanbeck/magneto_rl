#!/usr/bin/env python3
import sys
from magneto_env import MagnetoEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main ():
    # . Trying to learn SOMETHING
    path = '/home/steven/magneto_ws/outputs/'
    
    env = MagnetoEnv(render_mode="human", sim_mode="full", magnetic_seeds=10)
    rel_path = 'dqn/independent/multi_input/paraboloid_penalty/'
    
    # - Callback to save weights during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # - Loading specified weights
    model = DQN.load(path + 'weights/breakpoint.zip', env=env)
    
    # - Training
    try:
        for _ in range(1):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                env.render()
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                counter += 1
            env.close()
        
    finally:
        pass
        model.save(path + rel_path + 'breakpoint.zip')

if __name__ == "__main__":
    main()
