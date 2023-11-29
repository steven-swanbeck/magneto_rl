#!/usr/bin/env python3
import numpy as np
import math
# from geometry_msgs.msg import Pose
import random
import tqdm
from scipy.spatial.transform import Rotation as R

class Pose ():
    def __init__(self) -> None:
        self.position = Point()
        self.orientation = Quaternion()

class Point ():
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0

class Quaternion ():
    def __init__(self) -> None:
        self.w = 0
        self.x = 0
        self.y = 0
        self.z = 0

class MagnetoAction (object):
    def __init__(self, idx:int=None, pose:Pose=None, link_idx:str=None) -> None:
        self.idx = idx
        if pose is not None:
            self.pose = pose
        else:
            self.pose = Pose()
            self.pose.orientation.w = 1.0
        self.link_idx = link_idx
    
    def __repr__(self) -> str:
        return 'MagnetoAction()'
    
    def __str__(self) -> str:
        return f'idx: {self.idx}\npose: {self.pose}\nlink_idx: {self.link_idx}'

class MagnetoFootState (object):
    def __init__(self, state, link_id) -> None:
        self.link_id = link_id
        self.pose = state.pose
        self.magnetic_force = state.magnetic_force
    
    def __repr__(self) -> str:
        return 'MagnetoFootState()'
    
    def __str__(self) -> str:
        return f'link_id: {self.link_id}\npose: {self.pose}\nmagnetic_force: {self.magnetic_force}'
    
class MagnetoState (object):
    def __init__(self, state) -> None:
        self.ground_pose = state.ground_pose
        self.body_pose = state.body_pose
        self.foot0 = MagnetoFootState(state.AR_state, 'AR')
        self.foot1 = MagnetoFootState(state.AL_state, 'AL')
        self.foot2 = MagnetoFootState(state.BL_state, 'BL')
        self.foot3 = MagnetoFootState(state.BR_state, 'BR')
    
    def __repr__(self) -> str:
        return 'MagnetoState()'
    
    def __str__(self) -> str:
        return f'ground_pose: {self.ground_pose}\nbody_pose: {self.body_pose}\nfoot0: {self.foot0}\nfoot1: {self.foot1}\nfoot2: {self.foot2}\nfoot3: {self.foot3}'

def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z]) # in radians

def extract_ground_frame_positions (state:MagnetoState):
    r = R.from_quat([state.ground_pose.orientation.x, state.ground_pose.orientation.y, state.ground_pose.orientation.z, state.ground_pose.orientation.w])
    translation = np.expand_dims(np.array([state.ground_pose.position.x, state.ground_pose.position.y, state.ground_pose.position.z]), 1)
    T = np.linalg.inv(np.concatenate([np.concatenate([r.as_matrix(), translation], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0))
    
    pb = np.expand_dims(np.array([state.body_pose.position.x, state.body_pose.position.y, state.body_pose.position.z, 1]), 1)
    p0 = np.expand_dims(np.array([state.foot0.pose.position.x, state.foot0.pose.position.y, state.foot0.pose.position.z, 1]), 1)
    p1 = np.expand_dims(np.array([state.foot1.pose.position.x, state.foot1.pose.position.y, state.foot1.pose.position.z, 1]), 1)
    p2 = np.expand_dims(np.array([state.foot2.pose.position.x, state.foot2.pose.position.y, state.foot2.pose.position.z, 1]), 1)
    p3 = np.expand_dims(np.array([state.foot3.pose.position.x, state.foot3.pose.position.y, state.foot3.pose.position.z, 1]), 1)

    pb_ = T @ pb
    p0_ = T @ p0
    p1_ = T @ p1
    p2_ = T @ p2
    p3_ = T @ p3
    return {'body':pb_, 'feet': {0:p0_, 1:p1_, 2:p2_, 3:p3_}}

def return_closest (n, range):
    return min(range, key=lambda x:abs(x-n))
    
def iterate (env, num_steps=10, check_status=True):  
    env.reset()
    
    for i in range(num_steps):
        print(f'it: {i}')
        action = np.empty([3,], dtype=np.float64)
        action[0] = random.uniform(-1, 1)
        action[1] = random.uniform(-1, 1)
        action[2] = random.uniform(-1, 1)
        env.step(action, check_status=check_status)
    env.terminate_episode()

def global_to_body_frame (body_location, body_yaw, goal_location):
    r = np.empty((3, 3))
    r[0,:] = [np.cos(body_yaw), -np.sin(body_yaw), 0]
    r[1,:] = [np.sin(body_yaw), np.cos(body_yaw), 0]
    r[2,:] = [0, 0, 1]
    t = np.expand_dims(np.array([body_location[0], body_location[1], 0]), 1)
    T = np.linalg.inv(np.concatenate([np.concatenate([r, t], axis=1), np.expand_dims(np.array([0, 0, 0, 1]), 1).T], axis=0))
    g = np.expand_dims(np.array([goal_location[0], goal_location[1], 0, 1]), 1)
    g_b = T @ g
    return np.array([g_b[0][0], g_b[1][0]])

def body_to_global_frame (body_yaw, goal_location):
    r = np.empty((3, 3))
    r[0,:] = [np.cos(body_yaw), -np.sin(body_yaw), 0]
    r[1,:] = [np.sin(body_yaw), np.cos(body_yaw), 0]
    r[2,:] = [0, 0, 1]
    g = np.expand_dims(np.array([goal_location[0], goal_location[1], 0]), 1)
    g_b = r @ g
    return np.array([g_b[0][0], g_b[1][0]])

class paraboloid (object):
    def __init__(self, origin:np.array) -> None:
        self.w = origin[0]
        self.h = origin[1]
        
    def eval(self, location:np.array) -> float:
        return (location[0] - self.w)**2 + (location[1] - self.h)**2
    
class gaussian (object):
    def __init__(self, origin:np.array, sigma:float) -> None:
        self.x0 = origin[0]
        self.y0 = origin[1]
        self.sigma = sigma
    
    def eval (self, location:np.array) -> float:
        return (1 / (2 * np.pi * self.sigma**2)) * np.exp(-1 * ((location[0] - self.x0)**2 + (location[1] - self.y0)**2) / (2 * self.sigma**2))

class circle (object):
    def __init__(self, origin:np.array, radius:float) -> None:
        self.x0 = origin[0]
        self.y0 = origin[1]
        self.radius = radius
    
    def eval (self, location:np.array) -> float:
        if np.linalg.norm(location - np.array([self.x0, self.y0])) < self.radius:
            return -1
        else:
            return 0

class cone (object):
    def __init__(self, origin:np.array) -> None:
        self.w = origin[0]
        self.h = origin[1]
        
    def eval(self, location:np.array) -> float:
        return np.abs((location[0] - self.w)) + np.abs((location[1] - self.h))
