#!/usr/bin/env python3
# %%
import numpy as np
from magneto_utils import *
from magnetic_seeder import MagneticSeeder
import pygame
import cv2

class GamePlugin(object):
    
    # WIP
    def __init__(self, render_mode, render_fps, magnetic_seeds=10) -> None:
        self.link_idx = {
            'AR':0,
            'AL':1,
            'BL':2,
            'BR':3,
        }
        self.leg_reach = np.array([0.08, 0.35])
        self.wall_size = 5
        self.render_mode = render_mode
        self.fps = render_fps
        self.window = None
        self.clock = None
        self.wall_width = 5
        self.wall_height = 5
        self.im_width = 500
        self.im_height = 500
        self.window_size = 500
        self.scale = 500 / 5 #pixels/m
        self.heading_arrow_length = 0.2
        self.leg_length = 0.2
        self.body_radius = 0.08
        self.foot_radius = 0.03
        self.body_width = 0.2 #m
        self.body_width_pixels = self.scale * self.body_width
        self.body_height = 0.3 #m
        self.body_height_pixels = self.scale * self.body_height
        self.goal = np.array([1, 1]) # !
        self.heading = 0
        self.tolerable_foot_displacement = np.array([0.08, 0.35])
        self.tolerable_foot_angles = [
            np.array([-2.0944, 0.5236]),
            np.array([-0.5236, 2.0944]),
            np.array([1.0472, 3.6652]),
            np.array([2.6180, 5.23599]),
        ]
        
        self.mag_seeder = MagneticSeeder()
        self.num_seeds = magnetic_seeds
        self.foot_modified = False
    
    def report_state (self):
        return StateRep(self.ground_pose, self.body_pose, self.foot_poses, self.foot_mags)
    
    def update_goal (self, goal):
        self.goal = goal
    
    def verify_foot_position (self, id:int):
        # - if distance is too large or small, replace it with a new position that satisfies the constraints
        # - making sure angle is set in range
        self.foot_modified = False
        angle = np.arctan2(self.foot_poses[id].position.y, self.foot_poses[id].position.x)
        if (id == 2) or (id == 3):
            if angle < 0:
                angle = angle + 2 * np.pi
                self.foot_modified = True
        if angle < self.tolerable_foot_angles[id][0]:
            angle = self.tolerable_foot_angles[id][0]
            self.foot_modified = True
        elif angle > self.tolerable_foot_angles[id][1]:
            angle = self.tolerable_foot_angles[id][1]
            self.foot_modified = True
    
        # - making sure distance is in range
        norm = np.linalg.norm(np.array([self.foot_poses[id].position.x, self.foot_poses[id].position.y]), 2)
        if norm > self.tolerable_foot_displacement[1]:
            norm = self.tolerable_foot_displacement[1]
            self.foot_modified = True
        elif norm < self.tolerable_foot_displacement[0]:
            norm = self.tolerable_foot_displacement[0]
            self.foot_modified = True
        
        self.foot_poses[id].position.x = norm * np.cos(angle)
        self.foot_poses[id].position.y = norm * np.sin(angle)
    
    # WIP
    def update_action (self, link_id:str, pose:Pose) -> bool:
        update = body_to_global_frame(self.heading, np.array([pose.position.x, pose.position.y]))
        
        # print(f'Updating from: {self.foot_poses[self.link_idx[link_id]].position.x}, {self.foot_poses[self.link_idx[link_id]].position.y}')
        # print(f'to: {self.foot_poses[self.link_idx[link_id]].position.x + update[0]}, {self.foot_poses[self.link_idx[link_id]].position.y + update[1]}')
        self.foot_poses[self.link_idx[link_id]].position.x += update[0] # ? switching these to try to better correspond with the full sim
        self.foot_poses[self.link_idx[link_id]].position.y += update[1]
        
        # ! making legs not go past allowable bandwidth (disabling for leader-follower env)
        # self.verify_foot_position(self.link_idx[link_id])
        
        for ii in range(len(self.foot_mags)):
            self.foot_mags[ii] = self.mag_seeder.lookup_magnetism_modifier(np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y]))
        
        pos, heading = self.calculate_body_pose()
        self.body_pose.position.x = pos[0]
        self.body_pose.position.y = pos[1]
        self.body_pose.orientation.w = np.sin(heading)
        self.body_pose.orientation.z = np.cos(heading)
        # self.heading = heading # ! adding this back in has a huge impact on performance
    
    def begin_sim_episode (self) -> bool:
        self.ground_pose = Pose()
        self.ground_pose.orientation.w = 1.
        self.body_pose = Pose()
        self.body_pose.orientation.w = 1.
        self.foot_poses = [Pose(), Pose(), Pose(), Pose()]
        for ii in range(len(self.foot_poses)):
            self.foot_poses[ii].orientation.w = 1.
        self.foot_mags = np.array([1., 1., 1., 1.])
        self.heading = 0
        self.spawn_robot()
        _, self.heading = self.calculate_body_pose()
        self.raw_map, self.seed_locations, self.single_channel_map = self.mag_seeder.generate_map(self.num_seeds)
        self.game_background = self.mag_seeder.transform_image_into_pygame(self.raw_map)
        
    def paint_robot_and_goal (self, robot, goal):
        goal_pixels = self.mag_seeder.cartesian_to_image_coordinates(goal)
        robot_pixels = self.mag_seeder.cartesian_to_image_coordinates(robot)
        im_circ = cv2.circle(self.raw_map, goal_pixels, radius=20, color=(0, 255, 0), thickness=-1)
        im_goal = cv2.circle(im_circ, robot_pixels, radius=20, color=(100, 100, 100), thickness=-1)
        return im_goal

    def end_sim_episode (self) -> bool:
        # ? should this be here?
        # self.close()
        pass
    
    def reset_leg_positions (self):
        # - reset leg positions relative to it
        leg_angles = [7 * np.pi / 4, 1 * np.pi / 4, 3 * np.pi /4, 5 * np.pi / 4]
        for ii in range(len(self.foot_poses)):
            self.foot_poses[ii].position.x = self.body_pose.position.x + self.leg_length * np.cos(leg_angles[ii])
            self.foot_poses[ii].position.y = self.body_pose.position.y + self.leg_length * np.sin(leg_angles[ii])
    
    def _render_frame (self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA, 32)
        canvas = canvas.convert_alpha()

        body_center = self.cartesian_to_pygame_coordinates(np.array([self.body_pose.position.x, self.body_pose.position.y]))
        pygame.draw.circle(
            canvas,
            (150, 150, 150),
            center=body_center,
            radius=self.body_radius * self.scale * 2 / 3,
        )
        
        foot_pixel_positions = [self.cartesian_to_pygame_coordinates(np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y])) for ii in range(len(self.foot_poses))]
        for ii in range(len(self.foot_poses)):
            pygame.draw.circle(
                canvas,
                (150, 150, 150),
                center=foot_pixel_positions[ii],
                radius=self.foot_radius * self.scale,
            )
        
        heading_end = self.cartesian_to_pygame_coordinates(np.array([self.body_pose.position.x, self.body_pose.position.y]) + np.array([self.heading_arrow_length * np.cos(self.heading), self.heading_arrow_length * np.sin(self.heading)]))
        pygame.draw.line(
                canvas,
                (255, 255, 255),
                start_pos=body_center,
                end_pos=heading_end,
                width=3,
            )

        goal_center = self.cartesian_to_pygame_coordinates(self.goal)
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            center=goal_center,
            # radius=self.body_radius * self.scale / 2,
            radius=0.20 * self.scale, # & this should be set to the tolerance of the at_goal() function in the environment
        )
        
        if self.render_mode == "human":
            self.window.blit(pygame.surfarray.make_surface(self.game_background), (0, 0))
            self.window.blit(canvas, canvas.get_rect())
            
            myfont = pygame.font.SysFont("monospace", 15)
            for ii in range(len(self.foot_poses)):
                label = myfont.render(str(ii), 1, (255,255,0))
                self.window.blit(label, foot_pixel_positions[ii])
        
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
            
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def spawn_robot (self):
        self.body_pose.position.x = 0
        self.body_pose.position.y = 0
        
        leg_angles = [7 * np.pi / 4, 1 * np.pi / 4, 3 * np.pi /4, 5 * np.pi / 4]
        for ii in range(len(self.foot_poses)):
            self.foot_poses[ii].position.x += self.leg_length * np.cos(leg_angles[ii])
            self.foot_poses[ii].position.y += self.leg_length * np.sin(leg_angles[ii])
    
    # TODO think about way that the heading is calculated
    def calculate_body_pose (self):
        # . Pos is average position of the four feet
        # . Heading is perpendicular to
        
        px = np.mean([pose.position.x for pose in self.foot_poses])
        py = np.mean([pose.position.y for pose in self.foot_poses])
        
        pos = np.array([px, py])
        feet_pos = [[pose.position.x, pose.position.y] for pose in self.foot_poses]
        rel_feet_pos = [np.array(foot_pos) - pos for foot_pos in feet_pos]
        
        front_leg_v = rel_feet_pos[2] - rel_feet_pos[3]
        rear_leg_v = rel_feet_pos[1] - rel_feet_pos[0]
        
        theta_front = 0 - np.arctan2(front_leg_v[0], front_leg_v[1])
        theta_rear = 0 - np.arctan2(rear_leg_v[0], rear_leg_v[1])
        theta_average = (theta_front + theta_rear) / 2
        heading = theta_average# + np.pi / 2
        
        return pos, heading
    
    # TODO add magnetism here!
    def has_fallen (self):
        # - Check if outside map bounds
        if (np.abs(self.body_pose.position.x) > self.wall_size) or (np.abs(self.body_pose.position.y > self.wall_size)):
            return True
        
        # ! disabling this for now (kind of accounted for by progress constraints in update_action), so now only way to register fall is magnetic forces (or if robot is off wall)
        # # - Check if all feet are within spatial "bandwidth"
        # body_pos = np.array([self.body_pose.position.x, self.body_pose.position.y])
        # feet_pos = [np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y]) for ii in range(len(self.foot_poses))]
        # feet_pos_b = []
        
        # for ii in range(len(feet_pos)):
        #     feet_pos_b.append(body_to_global_frame(self.heading, feet_pos[ii] - body_pos))
        
        # if self.outside_bandwidth(feet_pos_b):
        #     return True
        
        if not self.verify_magnetic_integrity():
            return True
        
        return False
    
    def verify_magnetic_integrity (self):
        highest = np.delete(self.foot_mags, self.foot_mags.argmin())
        if np.sum(highest) < 2.:
            return False
        return True
    
    def outside_bandwidth (self, feet_pos_b):
        for ii in range(len(feet_pos_b)): 
            extension = np.linalg.norm(feet_pos_b[ii], 2)
            if extension > self.leg_reach[1]:
                return True
            if extension < self.leg_reach[0]:
                return True

            if ii == 0: # in fourth quadrant
                if not ((feet_pos_b[ii][0] >= 0) and (feet_pos_b[ii][1] <= 0)):
                    return True
            elif ii == 1: # in first quadrant
                if not ((feet_pos_b[ii][0] >= 0) and (feet_pos_b[ii][1] >= 0)):
                    return True
            elif ii == 2: # in second quadrant
                if not ((feet_pos_b[ii][0] <= 0) and (feet_pos_b[ii][1] >= 0)):
                    return True
            elif ii == 3: # in third quadrant
                if not ((feet_pos_b[ii][0] <= 0) and (feet_pos_b[ii][1] <= 0)):
                    return True
        return False
    
    def cartesian_to_pygame_coordinates (self, coords):
        output = np.array([
            coords[1] * (self.im_width / (2 * self.wall_width)) + self.im_width / 2,
            coords[0] * (self.im_height / (2 * self.wall_height)) + self.im_height / 2,
        ])
        return output

# . Compatability standins
class StateRep(object):
    def __init__(self, ground_pose, body_pose, foot_poses, foot_mags) -> None:
        self.ground_pose = ground_pose
        self.body_pose = body_pose
        self.AR_state = FootStateRep(foot_poses[0], foot_mags[0])
        self.AL_state = FootStateRep(foot_poses[1], foot_mags[1])
        self.BL_state = FootStateRep(foot_poses[2], foot_mags[2])
        self.BR_state = FootStateRep(foot_poses[3], foot_mags[3])
        
class FootStateRep(object):
    def __init__(self, pose, force) -> None:
        self.pose = pose
        self.magnetic_force = force
