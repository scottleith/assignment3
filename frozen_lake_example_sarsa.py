import numpy as np
import math
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

env = gym.make("FrozenLake-v0", map_name = "4x4")

def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        action = np.argmax( Q[state] )
    return action

def decay(parameter, step, decay_rate):
  return max(.001, min(parameter, 1. - math.log10((step + 1) / decay_rate)))

    
def SARSA(env, num_episodes, alpha, epsilon, gamma, epsilon_final, decay_rate):
  
  Q = defaultdict(lambda: np.random.random(4))
  episode_returns = []
  rolling_avg_100 = []
  goal = []
  goal_perc = []

  for episode in tqdm( range(1,num_episodes) ):
    state = env.reset()
    t = 0
    returns = 0
    epsilon *= decay_rate

    while True:
      action = epsilon_greedy_policy(Q, state, epsilon)
      next_state, reward, done, _ = env.step(action)
      next_state_type = env.desc.flatten()[int(next_state)]
      next_action = epsilon_greedy_policy(Q, next_state, epsilon)
      Q[state][action] += alpha * (reward + \
        gamma*Q[next_state][next_action] - Q[state][action])
      action = next_action
      state = next_state
      t += 1
      if done:
        break

    goal.append(next_state_type == b'G')
    episode_returns.append(reward)

    if episode > 100:
      goal_perc.append(np.mean(goal[-100:])*100)
      rolling_avg_100.append(np.mean(episode_returns[-100:]))

  return episode_returns, Q, rolling_avg_100, goal, goal_perc

er, q, avgs, g, gp = SARSA(
  env = env, num_episodes = 2000, alpha = 0.5, epsilon = 1., gamma = 1., 
  decay_rate = .99, epsilon_final = .05
  )

plt.plot(gp)
plt.plot(avgs)
