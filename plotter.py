import matplotlib.pyplot as plt
import numpy as np

model_directory = "./model"
reward_file = f"{model_directory}/reward_hist.txt"
plt.figure(figsize=(6,4), dpi=100)
plt.plot(np.loadtxt(reward_file, delimiter=" ", dtype=float))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()