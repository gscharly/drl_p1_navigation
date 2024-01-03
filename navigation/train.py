"""
Script that can be used to train a DQL agent to solve the Unity Bananas environment.
"""

import argparse
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from navigation.agent import DQNAgent

RANDOM_SEED = 10


def train_dqn(env: UnityEnvironment, brain_nm: str, agent: DQNAgent, weights_path: str,
              n_episodes: int = 2000, max_t: int = 1000,
              eps_start: float = 1.0, eps_end: float = 0.01,
              eps_decay: float = 0.995, solved_th: int = 13, n_episodes_score: int = 100):
    """Train a DQL agent

    Params
    ======
        env (UnityEnvironment): Unity environment
        brain_nm (str): brain name
        agent (DQNAgent): DQNAgent instance
        weights_path (str): path to save the weights to
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        solved_th (int): score for which the environment is considered solved
        n_episodes_score (int): number of episodes to compute the mean score for
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=n_episodes_score)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_nm]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_nm]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % n_episodes_score == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= solved_th:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            print(f'Saving weights to {weights_path}')
            torch.save(agent.qnetwork_local.state_dict(), weights_path)
            break
    return scores, agent


def plot_scores(scores: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path',
                        dest='env_path',
                        help='Unity environment local path')
    parser.add_argument('--episodes',
                        dest='episodes',
                        help='Number of episodes to train the agent',
                        default=2000,
                        type=int)
    parser.add_argument('--time-steps-per-episode',
                        dest='max_t',
                        help='Time steps per training episode',
                        default=1000,
                        type=int)
    parser.add_argument('--weights-path',
                        dest='weights_path',
                        help='Path to store the agents NN weights',
                        default='../weights/agent_weights.pth')
    parser.add_argument('--eps-start',
                        dest='eps_start',
                        help='Initial value of epsilon',
                        default=1.0,
                        type=float)
    parser.add_argument('--eps-end',
                        dest='eps_end',
                        help='Final value of epsilon',
                        default=0.01,
                        type=float)
    parser.add_argument('--eps-decay',
                        dest='eps_decay',
                        help='Epsilon decay factor',
                        default=0.995,
                        type=float)
    parser.add_argument('--gamma',
                        dest='gamma',
                        help='Discount factor',
                        default=0.99,
                        type=float)
    parser.add_argument('--learning-rate',
                        dest='lr',
                        help='NN learning rate',
                        default=5e-4,
                        type=float)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='Experince replay buffer size',
                        default=64,
                        type=int)
    args = parser.parse_args()
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init agent
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    dq_agent = DQNAgent(state_size=len(environment_info.vector_observations[0]),
                        action_size=brain.vector_action_space_size, seed=RANDOM_SEED,
                        discount_factor=args.gamma, lr=args.lr)
    # Train agent
    result_scores, trained_agent = train_dqn(env=unity_env, brain_nm=brain_name, agent=dq_agent,
                                             weights_path=args.weights_path,
                                             n_episodes=args.episodes, max_t=args.max_t,
                                             eps_start=args.eps_start, eps_end=args.eps_end,
                                             eps_decay=args.eps_decay)
    plot_scores(result_scores)
