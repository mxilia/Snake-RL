# Snake-RL
Deep-Q-Learning with various optimization technique implemented using pytorch to try and solve a 10x10 game of snake. The optimizations I use are Double Deep-Q-Learning, Dueling Structure and Noisy Network. In the future, I might also implement more optimizations or different reinforcement learning algorithms.
## Usage
To clone this repository, run:
```
git clone https://github.com/mxilia/Snake-RL.git
```
then:
```
cd Snake-RL
```
To train a new model, run:
```
python main.py -option 1 -modelName <Name>
```
To add optimization, use the following arguments:
<table>
    <tr>
        <th>Argument</th>
        <th>Info</th>
        <th>Default</th>
        <th>Action</th>
    </tr>
    <tr>
        <td>-dueling</td>
        <td>Passing this argument will make your model ultilise Dueling Structure.</td>
        <td>False</td>
        <td>Store true</td>
    </tr>
    <tr>
        <td>-double</td>
        <td>Passing this argument will make your model a Double DQN.</td>
        <td>False</td>
        <td>Store true</td>
    </tr>
    <tr>
        <td>-noisy</td>
        <td>Passing this argument makes your model ultilise Noisy Network for randomness instead of epsilon-greedy.</td>
        <td>False</td>
        <td>Store true</td>
    </tr>
</table>

For example, if you want to make a Double Dueling DQN, you run:
```
python main.py -option 1 -modelName <Name> -dueling -double
```

To adjust the hyperparameters, use the following arguments:
<table>
    <tr>
        <th>Argument</th>
        <th>Info</th>
        <th>Default</th>
    </tr>
    <tr>
        <td>-updateType</td>
        <td>
            Indicate whether the model is using interval target update or soft update<br>
            0: Interval target update, 1: Soft update<br>
            Example: -updateType 0
        </td>
        <td>1</td>
    </tr>
    <tr>
        <td>-episode</td>
        <td>
            Has 2 purposes. If you're training the model (-option 1) then this is the amount of the episodes the model'll be training for.<br>
            If you're testing the model (-option 2) then this is the version model you're going to test.<br>
            Example: -episode 50000
        </td>
        <td>10000</td>
    </tr>
    <tr>
        <td>-epsilon</td>
        <td>
            the starting value of epsilon.<br>
            Example: -epsilon 0.90
        </td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>-epsilonDecay</td>
        <td>
            the rate of which epsilon is decaying.<br>
            Example: -epsilonDecay 0.99
        </td>
        <td>0.99999</td>
    </tr>
    <tr>
        <td>-epsilonMin</td>
        <td>
            the minimum value that epsilon can be.<br>
            Example: -epsilonMin 0.01
        </td>
        <td>0.02</td>
    </tr>
    <tr>
        <td>-discount</td>
        <td>
            the value of discount factor. the closer it is to 1, the more the model cares about the future reward.<br>
            Example: -discount 0.90
        </td>
        <td>0.99</td>
    </tr>
    <tr>
        <td>-lr</td>
        <td>
            the value of learning rate.<br>
            Example: -lr 0.01
        </td>
        <td>0.0001</td>
    </tr>
    <tr>
        <td>-batchSize</td>
        <td>
            the size of minibatch the model is sampling.<br>
            Example: -batchSize 64
        </td>
        <td>32</td>
    </tr>
    <tr>
        <td>-memorySize</td>
        <td>
            the size of the replay buffer.<br>
            Example: -memorySize 100000
        </td>
        <td>200000</td>
    </tr>
    <tr>
        <td>-targetInt</td>
        <td>
            the number of episode in between before copying online network params into target network params. (For -updateType 0)<br>
            Example: -targetInt 1000
        </td>
        <td>500</td>
    </tr>
    <tr>
        <td>-tau</td>
        <td>
            the rate of soft updating. (For -updateType 1)<br>
            Example: -tau 0.01
        </td>
        <td>0.005</td>
    </tr>
</table>

For example:
```
python main.py -option 1 -modelName <Name> -dueling -epsilonMin 0.01 -tau 0.0001 -discount 0.90
```

To adjust the environment, use the following arguments:
<table>
    <tr>
        <th>Argument</th>
        <th>Info</th>
        <th>Default</th>
    </tr>
    <tr>
        <td>-envCol</td>
        <td>
            the number of columns of the snake game grid.<br>
            Example: -envcol 6
        </td>
        <td>10</td>
    </tr>
    <tr>
        <td>-envRow</td>
        <td>
            the number of rows of the snake game grid.<br>
            Example: -envRow 6
        </td>
        <td>10</td>
    </tr>
    <tr>
        <td>-envPxSize</td>
        <td>
            the size of each grid in the snake game.<br>
            Note: this is only for display and changing anything here might glitch the game.<br>
            Example: -envPxSize 40
        </td>
        <td>20</td>
    </tr>
</table>

To change the environment, you just add these arguments into your command line:
```
python main.py -option 1 -modelName <Name> -dueling -envCol 5 -envRow 10
```

Here're the other useful arguments:
<table>
    <tr>
        <th>Argument</th>
        <th>Info</th>
        <th>Default</th>
    </tr>
    <tr>
        <td>-checkpoint</td>
        <td>
            the number of episodes in between before saving the model.<br>
            Example: -checkpoint 5000
        </td>
        <td>2000</td>
    </tr>
    <tr>
        <td>-option</td>
        <td>
            -option 0 (If you want to play the game yourself.)<br>
            -option 1 (If you want to train the model.)<br>
            -option 2 (If you want to test the model.)<br>
            -option 3 (if you want to plot the model's result.)
        </td>
        <td>None</td>
    </tr>
</table>

After training, to test the model, run:
```
python main.py -option 2 -modelName <Name> <Also specify the optimization you added.> -episode <TheVersionYouWantToTest>
```

For example, if you trained a Dueling DQN named "bob", you want to test it when it's at episode 12000 and the environment is 7x7, run:
```
python main.py -option 2 -modelName bob -dueling -episode 12000 -envCol 7 -envRow 7
```

To plot the model's result, run:
```
python main.py -option 3 -modelName <Name>
```

To play the game yourself, run:
```
python main.py -option 0
```

## Performance
todo
## Reference
todo