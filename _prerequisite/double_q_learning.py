import gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
input_dim = env.observation_space.n
output_dim = env.action_space.n
Q_table_1 = np.random.rand(input_dim, output_dim)
Q_table_2 = np.random.rand(input_dim, output_dim)

num_episode = 1000
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.005
discount = 0.999
learning_rate = 0.1
reward_hist = []

for i in range(num_episode):
    state = env.reset()[0]
    while(True):
        optimal_action = np.argmax((Q_table_1+Q_table_2)[state])
        random_action = np.random.randint(0,4)
        action = np.random.choice([optimal_action, random_action], p=[1.0-epsilon, epsilon])
        next_state,reward,done,_,__ = env.step(action)
        if(reward==0 and done): reward=-1.0
        rand_table = np.random.choice([0, 1], p=[0.5, 0.5])
        if(rand_table == 0): 
            table_1_next_optimal = np.argmax(Q_table_1[next_state])
            Q_table_1[state][action]+=learning_rate*(reward+discount*Q_table_2[next_state][table_1_next_optimal]-Q_table_1[state][action])
        else:
            table_2_next_optimal = np.argmax(Q_table_2[next_state])
            Q_table_2[state][action]+=learning_rate*(reward+discount*Q_table_1[next_state][table_2_next_optimal]-Q_table_2[state][action])
        state = next_state
        epsilon=max(epsilon_min, epsilon*epsilon_decay)
        if(done):
            print(f"ep {i+1}:", reward)
            reward_hist.append(reward)
            break

# Test
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state = env.reset()[0]
while True:
    optimal_action = np.argmax((Q_table_1+Q_table_2)[state])
    next_state,reward,done,_,__ = env.step(optimal_action)
    state = next_state
    env.render()
    if(done): break