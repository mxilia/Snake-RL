import numpy as np
from test import test

def evaluate(agent, env, num_game=500):
    reward_list, score_list = [], []
    for i in range(num_game):
        reward, score = test(agent, env, 0)
        reward_list.append(reward)
        score_list.append(score)
        env.reset()
    score_list = np.array(score_list)
    score_mean = score_list.mean()
    score_max = score_list.max()
    score_min = score_list.min()
    print("Evaluation completed.\n---------------------")
    print(f"Average Score: {score_mean}")
    print(f"Max Score: {score_max}")
    print(f"Min Score: {score_min}")
    print("---------------------")
    return