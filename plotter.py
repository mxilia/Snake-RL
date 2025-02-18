import matplotlib.pyplot as plt
import numpy as np

model_directory = "./model"
epoch = int(open(f"{model_directory}/epoch.txt", "r").read())
epoch_folder = f"epoch_{epoch}"
reward_file = f"{model_directory}/{epoch_folder}/reward_list.txt"
plt.figure(figsize=(6,4), dpi=100)
plt.plot(np.loadtxt(reward_file, delimiter=" ", dtype=float))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()