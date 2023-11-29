# Reinforcement Learning for Traversal of Uncertain Vertical Terrain using a Magnetic Wall-Climbing Robot



## About
This project aims to give a quadrupedal magnetic-wall-climbing robot the ability to safely navigate a wall with randomly-seeded areas of weakened magnetism (meant to simulate heterogeneous material) to a goal position without falling.  

![](assets/goal.png)

Falling due to weak magnetism |  Avoiding areas of weak magnetism to reach goal
:-------------------------:|:-------------------------:
![](assets/falling.gif)  |  ![](assets/walking.gif)

## Contents
```rl/``` contains plugins a custom Gym environment for a Nexxis Magneto wall climbing robot with a ROS-based plugin to interface with an adapted version of the simulation [originally developed by Jee-Eun Lee](https://github.com/jeeeunlee/ros-pnc.git), a simplified game-based simulation for training and visualization purposes, utilities to create maps of magnetic variability, and training and evaluation scripts to learn from both full- and game-based simulations using Deep Q-learning.

Please refer to the [modified version of Jee-Eun's original simulation](https://github.com/steven-swanbeck/magneto-pnc.git) and to the [original project workspace](https://github.com/steven-swanbeck/magneto_rl_basic.git) to see more details of implementation and to use other learning algorithms (namely simple, recurrent, and convolutional implementations of proximal policy optimization) and approaches.  

## Requirements, Dependencies, and Building
These packages are built and tested on a system running ROS1 noetic on Ubuntu 20.04. However, the game simulation and all learning components are pure Python, and can be run independently of ROS or of the dependencies of the full simulation.

### Game Simulation
1. Clone the repository without submodules:
```
git clone https://github.com/steven-swanbeck/magneto_rl.git
```

### Full Simulation
1. Create a Catkin workspace:
```
mkdir -p catkin_ws/src && cd catkin_ws
```
2. Clone the contents of this repository:
```
git clone --recurse-submodules git@github.com:steven-swanbeck/magneto-rl.git src/
```

3. Install all package dependencies:
```
rosdep update
```
```
rosdep install --from-paths src --ignore-src -r -y
```
```
source magneto-pnc/install_sim.sh
```
Please refer to magneto-pnc/notes.txt for helpful installation and troubleshooting tips.

4. Build and source the workspace
```
catkin_make -DCMAKE_BUILD_TYPE=Release
```
```
source devel/setup.bash
```

## Running the Code
To train/test using the full simulation, run

```
roslaunch magneto_rl/base.launch
```
To train or evaluate using only the simple simulation, run
```
python3 train.py
```
or
```
python3 eval.py
```
respectively, from within ```rl/src/```.

PPO  |  DQN
:-------------------------:|:-------------------------:


![](assets/ppo.mp4)  |  ![](assets/dqn.mp4)

Notice in the results above that the PPO policy often fails to repsect and avoid areas of weakened magnetism, causing the robot to slip and sometimes fall. The DQN policy, in contrast, does a better job of avoiding the weak magnetism and - therefore staying on the wall - on its way to a goal position.