import sys
import inspect
import argparse

import train
import play as game
import environment as env
from model.ac_variant import *
from model.dqn_variant import *

classes = {name.lower(): cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)}
print(classes)
exit(0)

parser = argparse.ArgumentParser(description="parser")

parser.add_argument('-option', '--option', type=int, default=1,
                    help="Decides whether you\'re going to play, train or test. 0 for playing, 1 for training and 2 for testing.")

parser.add_argument('-modelType', '--model_type', type=str, required=True,
                    help='The model you\'re going to experiment with. (No space in between Ex: duelingdqn)')



args = parser.parse_args()
option = args.option
model_type = args.model_type.lower()

if __name__ == "__main__":
    if(option == 0): game.play()
    elif(option == 1): 
        pass
        train.train_a2c()
    elif(option == 2): pass

    else: 
        print("Invalid option.")
        exit(0)
