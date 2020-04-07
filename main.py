


import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make

from utils import *



def reward_coordination(obs, prev_obs):
    # prev_observationとobservationを比較して
    # 自分のstoneが連結しているかいなかでrewardを変更する。
    # 連結確認メソッド
    # import pdb; pdb.set_trace()

    obs_mat = np.array(obs.board).reshape(-1,7)
    prev_obs_mat = np.array(prev_obs.board).reshape(-1,7)
    new_stone_loc = np.where(obs_mat - prev_obs_mat == obs.mark)
    count_seq(new_stone_loc, obs_mat, obs.mark)

    return 0.5



def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    while not done:
        # Using epsilon-greedy to get an action
        action = TrainNet.get_action(observations, epsilon)

        # Caching the information of current state
        prev_observations = observations

        # Take action
        observations, reward, done, _ = env.step(action)

        # Apply new rules
        if done:
            if reward == 1: # Won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else: # Draw
                reward = 10
        else:
#             reward = -0.05 # Try to prevent the agent from taking a long move

            # Try to promote the agent to "struggle" when playing against negamax agent
            # as Magolor's (@magolor) idea
            reward = reward_coordination(observations, prev_observations) * 5

        rewards += reward

        # Adding experience into buffer
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)

        # Train the training model by using experiences in buffer and the target model
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            # Update the weights of the target model when reaching enough "copy step"
            TargetNet.copy_weights(TrainNet)
    return rewards


if __name__ == "__main__":
    env = ConnectX()
    gamma = 0.99
    copy_step = 25
    hidden_units = [128, 128, 128, 128, 128]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    epsilon = 0.5
    decay = 0.999
    min_epsilon = 0.01
    episodes = 50000
    precision = 7

    # prepare the agents

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    all_total_rewards = np.empty(episodes)
    all_avg_rewards = np.empty(episodes) # Last 100 steps
    all_epsilons = np.empty(episodes)

    # Initialize models
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    import copy

    pbar = tqdm(range(episodes))
    for n in pbar:
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        all_total_rewards[n] = total_reward
        avg_reward = all_total_rewards[max(0, n - 100):(n + 1)].mean()
        all_avg_rewards[n] = avg_reward
        all_epsilons[n] = epsilon
        
        pbar.set_postfix({
            'episode reward': total_reward,
            'avg (100 last) reward': avg_reward,
            'epsilon': epsilon
        })
        
        if n % 5000 == 0:
            TrainNet_adversarial = copy.deepcopy(TrainNet)
            env = ConnectX(switch_prob=0, pair=[None, TrainNet_adversarial])
            range_st = n//5000
            range_ed = range_st + 5000
            plt.plot(all_avg_rewards[range_st:range_ed])
            plt.xlabel('Episode')
            plt.ylabel('Avg rewards (100)')
            plt.show()
    plt.plot(all_avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Avg rewards (100)')
    plt.show()
    TrainNet.save_weights('./weights.pth')
