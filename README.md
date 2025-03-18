# Snake-RL
Deep-Q-Learning with various optimization technique implemented using pytorch to try and solve a 10x10 game of snake.
## Usage
To clone this repository, run:
```
git clone https://github.com/mxilia/Snake-RL.git
```
Then:
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
        <th>-dueling</th>
        <th>Passing this argument will make your model ultilise Dueling Structure.</th>
        <th>False</th>
        <th>Store true</th>
    </tr>
    <tr>
        <th>-double</th>
        <th>Passing this argument will make your model a Double DQN.</th>
        <th>False</th>
        <th>Store true</th>
    </tr>
    <tr>
        <th>Argument</th>
        <th>Passing this argument makes your model ultilise Noisy Network for randomness instead of epsilon-greedy.</th>
        <th>False</th>
        <th>Store true</th>
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
        <th>-updateType</th>
        <th>
            Indicate whether the model is using interval target update or soft update<br>
            0: Interval target update, 1: Soft update
            Example: -updateType 0
        </th>
        <th>1</th>
    </tr>
    <tr>
        <th>-episode</th>
        <th>
            Has 2 purposes. If you're training the model (-option 1) then this is the amount of the episodes the model'll be training for.<br>
            If you're testing the model (-option 2) then this is the version model you're going to test.
            Example: -episode 50000
        </th>
        <th>10000</th>
    </tr>
    <tr>
        <th>-epsilon</th>
        <th>
            The starting value of epsilon.
            Example: -epsilon 0.90
        </th>
        <th>1.0</th>
    </tr>
    <tr>
        <th>-epsilonDecay</th>
        <th>
            The rate of which epsilon is decaying.
            Example: -epsilonDecay 0.99
        </th>
        <th>0.99999</th>
    </tr>
    <tr>
        <th>-epsilonMin</th>
        <th>
            The minimum value that epsilon can be.
            Example: -epsilonMin 0.01
        </th>
        <th>0.02</th>
    </tr>
    <tr>
        <th>-discount</th>
        <th>
            The value of discount factor. The closer it is to 1, the more the model is caring about the future reward.
            Example: -discount 0.90
        </th>
        <th>0.99</th>
    </tr>
    <tr>
        <th>-lr</th>
        <th>
            The value of learning rate.
            Example: -lr 0.01
        </th>
        <th>0.0001</th>
    </tr>
    <tr>
        <th>-batchSize</th>
        <th>
            The size of minibatch the model is sampling.
            Example: -batchSize 64
        </th>
        <th>32</th>
    </tr>
    <tr>
        <th>-memorySize</th>
        <th>
            The size of the replay buffer.
            Example: -memorySize 100000
        </th>
        <th>200000</th>
    </tr>
    <tr>
        <th>-targetUpdateInterval</th>
        <th>
            The number of episode in between before copying online network params into target network params. (For -updateType 0)
            Example: -targetUpdateInterval 1000
        </th>
        <th>500</th>
    </tr>
    <tr>
        <th>-tau</th>
        <th>
            The rate of soft updating. (For -updateType 1)
            Example: -tau 0.01
        </th>
        <th>0.005</th>
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
        <th>-envCol</th>
        <th>
            The number of columns of the snake game grid.
            Example: -envcol 6
        </th>
        <th>10</th>
    </tr>
    <tr>
        <th>-envRow</th>
        <th>
            The number of rows of the snake game grid.
            Example: -envRow 6
        </th>
        <th>10</th>
    </tr>
    <tr>
        <th>-envPixelSize</th>
        <th>
            The size of each grid in the snake game.<br>
            Note: This is only for display and changing anything here might glitch the game.
            Example: -envPixelSize 40
        </th>
        <th>20</th>
    </tr>
</table>
To change the environment, you just add these arguments into your command line:
```
python main.py -option 1 -modelName <Name> -dueling -envCol 5 -envRow 10
```

## Performance
todo
## Reference
todo