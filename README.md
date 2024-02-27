# Udacity Deep Reinforcement Learning course - Value-based methods - P1 Navigation
This repository contains code that train an agent to solve the environment proposed in the Value-based methods section
of the Udacity Deep Reinforcement Learning (DRL) course.

# Environment

![Alt Text](./ipynb/agent.gif)

The environment consists of a single agent that has to pick up yellow bananas while avoiding blue bananas. A reward
of +1 is provided for collecting yellow bananas, and a reward of -1 for blue ones.

The state space has 37 dimensions that describe the agent's speed and object perception around the agent's forward
direction. Given this information, the agent has to select one of 4 discrete actions: move forward, move backward,
turn left or turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100
consecutive episodes.

# Getting started

## Unity environment
Unity doesn't need to be installed since the environment is already available. It can be downloaded from the following
links:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

When executing the training script, this path should be referenced with the --env-path argument.

## Python dependencies
The project uses Python 3.6 and relies on the [Udacity Value Based Methods repository](https://github.com/udacity/Value-based-methods#dependencies).
This repository should be cloned, and the instructions on the README should be followed to install the necessary
dependencies.

# Instructions
The repository contains 2 scripts under the navigation package: train.py and play.py.

## Train
The script train.py can be used to train the agent. It accepts the following arguments:
- env-path: path pointing to the Unity Bananas environment
- weights-path: path where the agent's NN weights will be stored
- episodes: number of episodes the agent should be trained for
- time-steps-per-episode: timesteps per episode
- eps-start: starting value for epsilon
- eps-end: minimum value for epsilon
- eps-decay: decay factor for epsilon
- gamma: discount rate
- learning-rate: agent's NN learning rate
- batch-size: size of the agent's experience replay buffer

Example: 
```
python train.py --env-path /home/carlos/cursos/udacity_rl_2023/repos/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64
--weights-path /home/carlos/cursos/udacity_rl_2023/projects/drl_p1_navigation/weights/agent_weights.pth
-- episodes 600
```

## Play
A trained agent can be used to play! To do so, the play.py script can be used, providing the Unity environment and
the agent's weights paths:

```
python play.py --env-path /home/carlos/cursos/udacity_rl_2023/repos/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64
--weights-path /home/carlos/cursos/udacity_rl_2023/projects/drl_p1_navigation/weights/agent_weights.pth
```

