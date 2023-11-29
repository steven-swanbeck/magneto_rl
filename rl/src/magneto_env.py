#!/usr/bin/env python3

# TODO
# - try independent leg motions
# - reduce paraboloid reward
# - add negative penalty for failing to make progress (with independent foot motions)

# %%
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
try:
    from magneto_ros_plugin import MagnetoRLPlugin
except ImportError:
    print("Unable to import ROS-based plugin!")
from magneto_utils import *
from magneto_game_plugin import GamePlugin
from copy import deepcopy
import pyscreenshot as ImageGrab
import moviepy.video.io.ImageSequenceClip
from datetime import datetime
import csv
from copy import deepcopy

class MagnetoEnv (Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps":10}
    # metadata = {"render_modes":["human", "rgb_array"], "render_fps":1}
    
    def __init__ (self, render_mode=None, sim_mode="full", magnetic_seeds=10, anneal=False):
        super(MagnetoEnv, self).__init__()
        
        self.sim_mode = sim_mode
        if self.sim_mode == "full":
            self.plugin = MagnetoRLPlugin(render_mode, self.metadata["render_fps"], magnetic_seeds)
        else:
            self.plugin = GamePlugin(render_mode, self.metadata["render_fps"], magnetic_seeds)
            self.render_mode = render_mode
        
        self.step_action_discretization = 7
        self.x_step = {0:-0.08, 1:-0.04, 2:-0.02, 3:0.0, 4:0.02, 5:0.04, 6:0.08}
        self.y_step = {0:-0.08, 1:-0.04, 2:-0.02, 3:0.0, 4:0.02, 5:0.04, 6:0.08}
        self.action_space = spaces.Discrete(self.step_action_discretization**2)
        
        self.observation_space = spaces.Dict({
            'goal': spaces.Box(low=-10, high=10, shape=(2,)),
            'magnetism': spaces.Box(low=0, high=1, shape=(4,)),
        })
        
        self.link_idx_lookup = {0:'AR', 1:'AL', 2:'BL', 3:'BR'}
        
        self.max_timesteps = 3000
        
        self.state_history = []
        self.action_history = []
        self.is_episode_running = False
        self.screenshots = []
        
        self.anneal = anneal
        self.use_temporary_goal = False
        self.temporary_goal_step_count = 0
        
        # *******************************************************************************************
        # # PPO STUFF
        # act_low = np.array([-1, -1, -1])
        # act_high = np.array([1, 1, 1])
        # self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        
        # obs_low = np.array([
        #     -10, -10, 0, 0, 0, 0,
        # ])
        # obs_high = np.array([
        #     10, 10, 1, 1, 1, 1
        # ])
        # self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # self.max_foot_step_size = 0.08 # ! remember this is here!
        # *******************************************************************************************
    
    def step (self, gym_action):
        self.state_history.append(MagnetoState(deepcopy(self.plugin.report_state())))
        
        # . Converting action from Gym format to one used by ROS plugin and other class members
        action = self.gym_2_action(gym_action)
        self.action_history.append(action)
        
        # . Taking specified action
        walk_order = np.random.permutation([0, 1, 2, 3])
        success = self.plugin.update_action(self.link_idx_lookup[walk_order[0]], action.pose)
        success = self.plugin.update_action(self.link_idx_lookup[walk_order[1]], action.pose)
        success = self.plugin.update_action(self.link_idx_lookup[walk_order[2]], action.pose)
        success = self.plugin.update_action(self.link_idx_lookup[walk_order[3]], action.pose)
        
        # *******************************************************************************************
        # # PPO STUFF
        # # . Taking specified action
        # success = self.plugin.update_action(self.link_idx_lookup[action.idx], action.pose)
        # # .make permutation of remaining three legs
        # legs = np.delete(np.array([0, 1, 2, 3]), action.idx)
        # walk_order = np.random.permutation(legs)
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[0]], action.pose)
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[1]], action.pose)
        # success = self.plugin.update_action(self.link_idx_lookup[walk_order[2]], action.pose)
        # *******************************************************************************************
        
        # . Observation and info
        obs_raw = self._get_obs(format='ros')
        info = self._get_info()
        
        # . Simulated annealing
        if self.anneal:
            self.monitor_progress()
        
        # . Termination determination
        reward, is_terminated = self.calculate_reward(obs_raw, action)
        # reward, is_terminated = self.calculate_reward(obs_raw, action, strategy='cone')
        
        # .Converting observation to format required by Gym
        obs = self.state_2_gym(obs_raw)
        
        if self.sim_mode == "full":
            print('-----------------')
            print(f'Step reward: {reward}')
            print(f'Distance from goal: {np.linalg.norm(self.goal - np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y]), 1)}')
            print(f'Body goal: {self.goal}')
            print(f'Body position: {np.array([obs_raw.body_pose.position.x, obs_raw.body_pose.position.y])}')
            print(f'Obs: {obs}')
            print('-----------------')
        self.timesteps += 1
        
        truncated = False
        if self.timesteps > self.max_timesteps:
            truncated = True
            is_terminated = True
        
        return obs, reward, is_terminated, False, info
    
    def calculate_reward (self, state, action, strategy="paraboloid"):
        is_terminated:bool = False
        
        if self.has_fallen(state):
            is_terminated = True
            reward = -1000
        elif self.at_goal(state, 0.5):
            is_terminated = True
            # reward = 1000
            reward = 100000
        else:
            if strategy == "paraboloid":
                curr = np.array([state.body_pose.position.x, state.body_pose.position.y])
                
                paraboloid_scaling_factor = 0.03
                if self.use_temporary_goal:
                    reward = -1 * paraboloid_scaling_factor * self.temporary_reward_paraboloid.eval(curr)
                else:
                    reward = -1 * paraboloid_scaling_factor * self.reward_paraboloid.eval(curr)
                
                gaussian_scaling_factor = 0.5
                for ii in range(len(self.reward_gaussians)):
                    reward += -1 * gaussian_scaling_factor * self.reward_gaussians[ii].eval(curr)
            
            elif strategy == "cone":
                curr = np.array([state.body_pose.position.x, state.body_pose.position.y])

                cone_scaling_factor = 0.025
                reward = -1 * cone_scaling_factor * self.reward_cone.eval(curr)
                
                gaussian_scaling_factor = 1.0
                for ii in range(len(self.reward_gaussians)):
                    reward += -1 * gaussian_scaling_factor * self.reward_gaussians[ii].eval(curr)
        
        return reward, is_terminated
    
    def proximity_reward (self, state, action, multipliers):
        if len(self.state_history) < 1:
            return 0
        
        proximity_change = self.calculate_distance_change(state, action)
        
        if proximity_change > 0:
            return proximity_change * multipliers[1]
        return proximity_change * multipliers[0]
    
    def get_state_history (self):
        return self.state_history
    
    def calculate_distance_change (self, state, action):
        if action.idx == 0:
            foot_pos = np.array([state.foot0.pose.position.x, state.foot0.pose.position.y])
            prev_foot_pos = np.array([self.state_history[-1].foot0.pose.position.x, self.state_history[-1].foot0.pose.position.y])
        elif action.idx == 1:
            foot_pos = np.array([state.foot1.pose.position.x, state.foot1.pose.position.y])
            prev_foot_pos = np.array([self.state_history[-1].foot1.pose.position.x, self.state_history[-1].foot1.pose.position.y])
        elif action.idx == 2:
            foot_pos = np.array([state.foot2.pose.position.x, state.foot2.pose.position.y])
            prev_foot_pos = np.array([self.state_history[-1].foot2.pose.position.x, self.state_history[-1].foot2.pose.position.y])
        elif action.idx == 3:
            foot_pos = np.array([state.foot3.pose.position.x, state.foot3.pose.position.y])
            prev_foot_pos = np.array([self.state_history[-1].foot3.pose.position.x, self.state_history[-1].foot3.pose.position.y])
        
        body_pos = np.array([state.body_pose.position.x, state.body_pose.position.y])
        prev_body_pos = np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y])
        
        prev_body_goal = self.goal - prev_body_pos
        prev_foot_goal = self.goal - prev_foot_pos
        
        curr_body_goal = self.goal - body_pos
        curr_foot_goal = self.goal - foot_pos
        
        prev_proj = np.dot(prev_foot_goal, prev_body_goal)
        curr_proj = np.dot(curr_foot_goal, curr_body_goal)
        
        prev_dist = np.linalg.norm(prev_body_pos, 2)
        curr_dist = np.linalg.norm(body_pos, 2)
        
        return prev_dist - curr_dist
    
    def screenshot (self):
        self.screenshots.append(np.array(ImageGrab.grab(bbox=(100, 200, 1800, 1050))))

    def export_video (self, fps=10):
        stamp = str(datetime.now())
        
        if len(self.screenshots) > 0:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(self.screenshots, fps=fps)
            clip.write_videofile('/home/steven/magneto_ws/outputs/single_walking/' + stamp + '.mp4')
            
            fields = [stamp, str(self.timesteps), str(self.goal[0]), str(self.goal[1]), str(self.state_history[-1].body_pose.position.x), str(self.state_history[-1].body_pose.position.y)]
            with open(r'/home/steven/magneto_ws/outputs/single_walking/log.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        
        return stamp
    
    def _get_obs (self, format='gym'):
        state = MagnetoState(self.plugin.report_state())
        if format == 'gym':
            return self.state_2_gym(state)
        return state
    
    def _get_info (self):
        return {}

    def render (self):
        self.plugin._render_frame()
    
    def reset (self, seed=None, options=None):
        super().reset(seed=seed)
        if self.is_episode_running:
            self.terminate_episode()
        self.begin_episode()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def begin_episode (self) -> bool:
        self.state_history = []
        self.action_history = []
        self.is_episode_running = True
        self.timesteps = 0
        self.goal = np.array([random.uniform(-4.5, 4.5),random.uniform(-4.5, 4.5)])
        self.reward_paraboloid = paraboloid(self.goal)
        self.reward_cone = cone(self.goal)
        self.plugin.update_goal(self.goal)
        self.plugin.begin_sim_episode()
        self.reward_gaussians = []
        for ii in range(len(self.plugin.seed_locations)):
            self.reward_gaussians.append(circle(self.plugin.seed_locations[ii], 0.2)) #0.6
        self.single_channel_map = self.plugin.single_channel_map
        self.use_temporary_goal = False
        return True

    def terminate_episode (self) -> bool:
        self.is_episode_running = False
        self.temporary_goal_step_count = 0
        self.export_video()
        return self.plugin.end_sim_episode()
    
    def close (self):
        self.is_episode_running = False
        return self.terminate_episode()

    def has_fallen (self, state, tol_pos=0.18, tol_ori=1.2):
        if self.making_insufficient_contact(state) == 4:
            return True
        if self.sim_mode != "full":
            return self.plugin.has_fallen()
        return False
    
    def making_insufficient_contact (self, state, tol=0.002):
        positions = extract_ground_frame_positions(state)
        insufficient = 0
        error_msg = 'Robot is making insufficient contact at:'
        for key, value in positions['feet'].items():
            if value[2][:] > tol: # z-coordinate
                error_msg += f'\n   Foot {key}! Value is {value[2][0]} and allowable tolerance is set to {tol}.'
                insufficient += 1
        if insufficient > 0:
            print(error_msg)
        return insufficient

    def at_goal (self, obs, tol=0.20):
        if np.linalg.norm(np.array([obs.body_pose.position.x, obs.body_pose.position.y]) - self.goal) < tol:
            return True
        return False
    
    def foot_at_goal (self, obs, tol=0.01):
        if np.linalg.norm(np.array([obs.foot0.pose.position.x, obs.foot0.pose.position.y]) - self.goal) < tol:
                return True
        return False

    def get_foot_from_action (self, x):
        # - one-hot encoded values
        # return np.argmax(x)
        # - single continuous value
        if x > 0.5:
            return 3
        if x > 0.0:
            return 2
        if x < -0.5:
            return 0
        return 1

    def gym_2_action (self, gym_action:np.array) -> MagnetoAction:
        action = MagnetoAction()
        
        action.pose.position.x = self.x_step[int(gym_action / self.step_action_discretization)]
        action.pose.position.y = self.y_step[gym_action % self.step_action_discretization]
        
        # *******************************************************************************************
        # # PPO STUFF
        # action.idx = self.get_foot_from_action(gym_action[0])
        # action.pose.position.x = self.max_foot_step_size * gym_action[1]
        # action.pose.position.y = self.max_foot_step_size * gym_action[2]
        # *******************************************************************************************

        return action
    
    def state_2_gym (self, state:MagnetoState) -> np.array:
        _, _, body_yaw = euler_from_quaternion(state.body_pose.orientation.w, state.body_pose.orientation.x, state.body_pose.orientation.y, state.body_pose.orientation.z)
        
        if self.use_temporary_goal:
            relative_goal = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, self.temporary_goal)
        else:
            relative_goal = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, self.goal)
        
        relative_foot0 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot0.pose.position.x, state.foot0.pose.position.y]))
        relative_foot1 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot1.pose.position.x, state.foot1.pose.position.y]))
        relative_foot2 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot2.pose.position.x, state.foot2.pose.position.y]))
        relative_foot3 = global_to_body_frame(np.array([state.body_pose.position.x, state.body_pose.position.y]), body_yaw, np.array([state.foot3.pose.position.x, state.foot3.pose.position.y]))

        magnetic_forces = np.array([state.foot0.magnetic_force, state.foot1.magnetic_force, state.foot2.magnetic_force, state.foot3.magnetic_force])
        
        if self.sim_mode == "full":
            relative_goal = -1 * relative_goal
            relative_foot0 = -1 * relative_foot0
            relative_foot1 = -1 * relative_foot1
            relative_foot2 = -1 * relative_foot2
            relative_foot3 = -1 * relative_foot3
            
        
        gym_obs = {
            'goal': relative_goal,
            'magnetism': magnetic_forces,
        }
        
        # *******************************************************************************************
        # # PPO STUFF
        # gym_obs = np.concatenate((relative_goal, magnetic_forces), dtype=np.float32)
        # *******************************************************************************************
        
        return gym_obs
    
    def foot_closest_to_goal (self, state:MagnetoState, body_yaw=0):
        feet_pos = [
            np.array([state.foot0.pose.position.x, state.foot0.pose.position.y]),
            np.array([state.foot1.pose.position.x, state.foot1.pose.position.y]),
            np.array([state.foot2.pose.position.x, state.foot2.pose.position.y]),
            np.array([state.foot3.pose.position.x, state.foot3.pose.position.y]),
        ]
        relative_distances = np.zeros((4,), np.float32)
        for ii in range(len(feet_pos)):
            relative_distances[ii] = np.linalg.norm(global_to_body_frame(self.goal, body_yaw, feet_pos[ii]), 2)
        return np.argmin(relative_distances)
    
    def monitor_progress (self, num_actions=10, max_steps=50):
        # . if we haven't progressed to the goal by some amount within the past x turns, generate a new random goal, once we get there, try to go to the real goal again
        if len(self.state_history) < num_actions:
            return
        
        past = self.goal - np.array([self.state_history[-10].body_pose.position.x, self.state_history[-10].body_pose.position.y])
        curr = self.goal - np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y])
        
        if not self.use_temporary_goal:
            if np.linalg.norm(curr - past, 2) < 0.01:
                self.use_temporary_goal = True
                
                self.temporary_goal = np.array([self.state_history[-1].body_pose.position.x, self.state_history[-1].body_pose.position.y]) + np.array([random.uniform(-2, 2),random.uniform(-2, 2)])
                self.temporary_reward_paraboloid = paraboloid(self.temporary_goal)
                print(f'Switching to temporary goal at {self.temporary_goal}!')
        else:
            self.temporary_goal_step_count += 1
            if self.temporary_goal_step_count > max_steps:
                self.use_temporary_goal = False
                self.temporary_goal_step_count = 0
                print(f'Switching back to original goal at {self.goal}!')
