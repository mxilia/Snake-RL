import gym
import numpy as np

env = gym.make("Taxi-v3")
input_dim = env.observation_space.n
output_dim = env.action_space.n
Q_table = np.random.rand(input_dim, output_dim)

num_episode = 2000
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
discount = 0.99
learning_rate = 0.2
reward_hist = []

for i in range(num_episode):
    state = env.reset()[0]
    reward_ep = 0
    while(True):
        optimal_action = np.argmax(Q_table[state])
        random_action = np.random.randint(0, output_dim)
        action = np.random.choice([optimal_action, random_action], p=[1.0-epsilon, epsilon])
        next_state,reward,done,_,__ = env.step(action)
        reward_ep+=reward
        Q_table[state][action]+=learning_rate*(reward+discount*np.max(Q_table[next_state])-Q_table[state][action])
        state = next_state
        epsilon=max(epsilon_min, epsilon*epsilon_decay)
        if(done): break
    print(f"ep {i+1}:", reward_ep)
    reward_hist.append(reward_ep)
            
# Test
env = gym.make("Taxi-v3", render_mode="human")
state = env.reset()[0]
reward_ep = 0
while True:
    optimal_action = np.argmax(Q_table[state])
    next_state,reward,done,_,__ = env.step(optimal_action)
    reward_ep+=reward
    state = next_state
    env.render()
    if(done): break
print("Final reward:",reward_ep)