import matplotlib.pyplot as plt
import numpy as np

model_directory = "./double_dueling_dqn"
reward_file = f"{model_directory}/reward_hist.txt"
reward = list(np.loadtxt(reward_file, delimiter=" ", dtype=float))
list_plot = []
mode="mean"

if(mode=="mean"):
    sum = 0.0
    cnt = 0.0
    for e in reward:
        sum+=float(e)
        cnt+=1.0
        list_plot.append(sum/cnt)
elif(mode=="raw"): list_plot=reward

plt.figure(figsize=(6,4), dpi=100)
plt.plot(list_plot)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()