"""
Script that can be used to play the Unity Bananas environment with a pretrained agent.
"""

import argparse

import torch
from unityagents import UnityEnvironment

from navigation.agent import DQNAgent


def play_with_agent(env: UnityEnvironment, brain_nm: str, agent: DQNAgent):
    """
    Uses a pretrained agent to play a Unity environment.

    Params
    ======
        env (UnityEnvironment): Unity environment
        brain_nm (str): brain name
        agent (DQNAgent): DQNAgent instance
    """
    env_info = env.reset(train_mode=False)[brain_nm]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent.act(state)  # select an action
        env_info = env.step(action)[brain_nm]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path',
                        dest='env_path',
                        help='Unity environment local path')
    parser.add_argument('--weights-path',
                        dest='weights_path',
                        help='Path to store the agents NN weights')
    args = parser.parse_args()
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path, seed=50)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init agent
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    dqn_agent = DQNAgent(state_size=len(environment_info.vector_observations[0]),
                         action_size=brain.vector_action_space_size,
                         seed=10)
    # Load weights
    dqn_agent.qnetwork_local.load_state_dict(torch.load(args.weights_path))
    # Play!
    play_with_agent(unity_env, brain_name, dqn_agent)
