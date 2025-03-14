import matplotlib.pyplot as plt
import numpy as np

def get_avg(array):
    avg_list = []
    sum = 0.0; cnt = 0.0
    for e in array:
        sum+=float(e); cnt+=1.0
        avg_list.append(sum/cnt)
    return avg_list

checkpoint_path = "./checkpoints"
model_directory = f"{checkpoint_path}/dueling_ddqn_6x6"
score_file = f"{model_directory}/score_hist.txt"
reward_file = f"{model_directory}/reward_hist.txt"
score = list(np.loadtxt(score_file, delimiter=" ", dtype=float))
reward = list(np.loadtxt(reward_file, delimiter=" ", dtype=float))
fig, axes = plt.subplots(1, 3, figsize=(14,4), dpi=100)

axes[0].plot(reward)
axes[0].set_title = "Episode Reward"
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')

axes[1].plot(get_avg(reward))
axes[1].set_title = "Avg Reward"
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Avg Reward')

axes[2].plot(score)
axes[2].set_title = "Score"
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Score')

plt.show()