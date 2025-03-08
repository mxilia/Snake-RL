import sys
import shlex
import pygame
import inspect
import argparse

import test
import train
import play as Game
import environment as Env
from algorithm.ac_variant import *
from algorithm.dqn_variant import *

checkpoint_path = "./checkpoints"

models = {"a2c" : [None, 0], "dqn" : [None, 1], "doubledqn" : [None, 1]}
for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass): 
    if(name.lower() in models): models[name.lower()][0] = cls

parser = argparse.ArgumentParser(description="parser")

parser.add_argument('-option', '--option', type=int, required=True,
                    help='Decides whether you\'re going to play, train or test. 0 for playing, 1 for training and 2 for testing.')

parser.add_argument('-modelType', '--model_type', type=str, default="",
                    help='The model you\'re going to experiment with. (No space in between Ex: doubledqn)')

parser.add_argument('-modelName', '--model_name', type=str, default=None,
                    help='The name of your model.')

parser.add_argument('-noisy', '--noisy', action="store_true", default=False,
                    help='Decides whether you\'re making your dqn model noisy or not.')

parser.add_argument('-dueling', '--dueling', action="store_true", default=False,
                    help='Decides whether you\'re using normal structure or dueling structure. (True for dueling Structure, only works for dqn)')

parser.add_argument('-updateType', '--update_type', type=int, default=1,
                    help='Decides whether you\'re going to make update target network by either interval update or soft update (0 for interval update and 1 for soft update)')

parser.add_argument('-episode', '--num_episode', type=int, default=10000,
                    help='How many episodes you want to train the agent for.')

parser.add_argument('-epsilon', '--epsilon', type=float, default=1.0,
                    help='Starting value of epsilon.')

parser.add_argument('-epsilonDecay', '--epsilon_decay', type=float, default=0.99999,
                    help='The rate of epsilon decay.')

parser.add_argument('-epsilonMin', '--epsilon_min', type=float, default=0.02,
                    help='The minimum value that epsilon can get.')

parser.add_argument('-discount', '--discount', type=float, default=0.99,
                    help='The value of discount.')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='The value of learning rate. (For DQN)')

parser.add_argument('-batchSize', '--batch_size', type=int, default=32,
                    help='The size of minibatch.')

parser.add_argument('-memorySize', '--memory_size', type=int, default=200000,
                    help='The size of memory buffer.')

parser.add_argument('-targetUpdateInterval', '--target_net_update_int', type=int, default=500,
                    help='The number of episodes between each target network update. (For interval update)')

parser.add_argument('-tau', '--tau', type=float, default=0.005,
                    help='The value of tau. (For soft update)')

parser.add_argument('-lr_a', '--lr_a', type=float, default=0.00002,
                    help='The value of learning rate. (For Actor)')

parser.add_argument('-lr_c', '--lr_c', type=float, default=0.00002,
                    help='The value of learning rate. (For Critic)')

parser.add_argument('-envCol', '--env_col', type=int, default=10,
                    help='How many columns you want in your snake board.')

parser.add_argument('-envRow', '--env_row', type=int, default=10,
                    help='How many rows you want in your snake board.')

parser.add_argument('-envPixelSize', '--env_pixel_size', type=int, default=20,
                    help='How large you want each grid to be (Unit: px)')

args = parser.parse_args()
option = args.option
model_type = args.model_type.lower()
model_name = args.model_name
noisy = args.noisy
dueling = args.dueling
update_type = args.update_type
num_episode = args.num_episode
epsilon = args.epsilon
epsilon_decay = args.epsilon_decay
epsilon_min = args.epsilon_min
discount = args.discount
learning_rate = args.learning_rate
batch_size = args.batch_size
memory_size = args.memory_size
target_net_update_int = args.target_net_update_int
tau = args.tau
lr_a = args.lr_a
lr_c = args.lr_c
env_col = args.env_col
env_row = args.env_row
env_pixel_size = args.env_pixel_size

if __name__ == "__main__":
    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
    env_config = Env.GameConfig(env_col, env_row, env_pixel_size)
    env = Env.Game(env_config)
    input_dim = env.INPUT_SHAPE
    output_dim = env.OUTPUT_SHAPE

    if(option == 0):  Game.play(env), pygame.quit(), exit(0)
    if(option != 1 and option !=2): print("Invalid Option"), exit(0)
    if(model_type not in models): print("Invalid Model Type"), exit(0)

    agent = None
    if(models[model_type][1] == 0): # Actor Critic Variant
        agent = models[model_type][0](input_dim, output_dim, model_name=model_name)
        agent.set_value(num_episode, discount, lr_a, lr_c)
    else: # DQN Variant
        agent = models[model_type][0](input_dim, output_dim, noisy=noisy, dueling=dueling, soft_update=bool(update_type), model_name=model_name)
        agent.set_value(num_episode, epsilon, epsilon_decay, epsilon_min, discount, learning_rate, batch_size, memory_size, target_net_update_int, tau)
    
    if(option == 1):
        args_dir = f"{agent.get_directory()}/args.txt"
        with open(args_dir, "w") as f: f.write("python " + " ".join(shlex.quote(arg) for arg in sys.argv))
        if(models[model_type][1] == 0): train.train_a2c(agent, env)
        else: train.train_dqn(agent, env)
    else:
        print("Pick the episode you want to test.")
        agent.get_model(f"snake_ep_{num_episode}", False)
        agent.epsilon = 0
        if(models[model_type][1] == 0): test.test_a2c(agent, env)
        else: test.test_dqn(agent, env)
    pygame.quit()

# python main.py -option 1 -modelType doubledqn -dueling -noisy -modelName noisy_dueling_ddqn -episode 10000
# python main.py -option 1 -modelType doubledqn -dueling -modelName dueling_ddqn_6x6 -envCol 6 -envRow 6 -episode 50000 -epsilonMin 0.01 -discount 0.90
# python main.py -option 1 -modelType doubledqn -dueling -modelName dueling_ddqn_4x4 -envCol 4 -envRow 4 -episode 10000 -epsilonDecay 0.9999 -epsilonMin 0.01 -discount 0.95
# python main.py -option 2 -modelType doubledqn -dueling -modelName dueling_ddqn
# python main.py -option 2 -modelType doubledqn -dueling -modelName dueling_ddqn_4x4 -envCol 4 -envRow 4 -episode 10000
