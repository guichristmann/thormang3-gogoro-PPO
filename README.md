# Scooter control with PPO (NVIDIA Isaac Gym)

This repository contains the environment code and configuration files for the NVIDIA Isaac Gym simulator. The environment consists of a 3D model of a large-sized humanoid robot (THORMANG3) riding a Gogoro electric scooter. This work was done as part of my Master's thesis titled "Balance and Control of a Humanoid Robot on an Electric Scooter", where the RL model was compared to classical PID control.  

Agent being controlled with a joystick:

![joystick](Docs/joystick.gif)

### Training Details

The goal of the agent was to follow a target $\dot{\theta}$ command, that is an intended turning rate, such as 15 degrees per second. The agent's only action was turning the steering joint of the scooter.

The agent received more reward the closer it got to the target $\dot{\theta}$ according to the following equation:
$$
R = 1.0 - |\dot{\theta} - {\dot{\theta}}^{targ}|^\frac{1}{5},
$$
where $R$ is the reward, $\dot{\theta}$ is the current turning rate and $\dot{\theta}^{targ}$ is the target turning rate. This way, the agent received a maximum reward of $+1$ for perfectly matching the command. The agent was not explicitly rewarded for balancing, but was punished with a value of $-2$ for falling over (a tilt angle of over 60 degrees).

Observation space (12 observations, 1 being the command):

* 1 x Current steering angle;
* 1 x Rear wheel velocity;
* 3 x Orientation (roll, pitch, yaw);
* 3 x Angular velocities;
* 3 x Linear velocities;
* **Target $\dot{\theta}$ -- command.** 

Action space was 1: Set angle of steering joint.

During training, every 2.5 seconds (of simulation time) the target $\dot{\theta}$ command is offset by a random, uniformly sampled value between $-5.7$ and $+5.7$ degrees. Every time the environment is reset a new random velocity is set. 

The agent was trained with PPO using the NVIDIA Isaac Gym simulator. Training for ~6 million timesteps takes about 3 hours of real-time on a single RTX 2080 Super.

### Files

`Code/environment.py` contains the implementation of the environment. This file should be placed under the `rlgpu/tasks/` folder in the Isaac Gym simulator.

`Code/env_config.yaml` contains the training and visualization parameters for the environment. This file is loaded when the environment is created and should be placed under `rlgpu/cfg`.

`Code/training_config.yaml` the training-specific parameters (e.g. network size, learning rate, etc.). It should be placed under `rlgpu/cfg/train`.

`Assets/` contains the robot and scooter models and its URDF description.

For reference, under `Docs/` there's a copy of the presentation slides for my thesis.