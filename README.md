# DoubleDQN Implementation
Implementation of DoubleDQN based on the publication by deepmind ["Human-level control through deep reinforcement
learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). This implementation was done on two most popular atari games: Breakout and Space Invaders.
## Tensorboard
To look at the graphics of loss and mean reward per game, one can run the command:
For Breakout
```bash
tensorboard --logdir ./main_results/Tensorboard/Breakout/
```
For Space Invaders
```bash
tensorboard --logdir ./main_results/Tensorboard/SpaceInvaders/
```
## Best model
To view the results of training, first, chose the game name in parameters.py file, then run 'DQN testing.ipynb'.
