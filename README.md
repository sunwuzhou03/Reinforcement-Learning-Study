# rl_learning

## Snake-v0
### state space: 6
(fx-x, fy-y, left, right, up ,down)

(x, y) is the position of the snake head. 

(fx, fy) is the position of the food.    

left indicates whether the left side of the snake’s head is a boundary or part of the snake’s body. left can only be 0 or 1. 

...

### action space: 4
The snake can move in four directions: up, down, left, and right.

### Algorithm
#### Implemented algorithm
dqn, ddqn, drqn

reinforce, reinforce with baseline

ac, ac with target, a2c, a2c with target

ppo

#### To-be-implemented algorithm
dueling dqn

a3c

trpo

ddpg

sac

#### Successful results

!['ppo 20*20'](https://github.com/sunwuzhou03/rl_learning/gif/Snake-v1\PPO.gif)

 You can find more successful result in gif file.  

## CartPole-v0

You can just modify the env_name from 'Snake-v0' to 'CartPole-v0' and adjust the hyper-parameters so that you train.

The program include: ac, a2c, ac_target, a2c_target, ddqn, ddrqn, dqn, drqn, ppo, reinforce, reinforce_baseline. 

## Pendulum-v1

You can run the ppoPendulum.py program and train agent in this environment.

## BipedalWalker-v3 and BipedalWalkerHardcore-v3

You can just modify the env_name from 'BipedalWalker-v3' to 'BipedalWalkerHardcore-v3' in the program sacwalker.py so that you train the agent in two environment.

### The results

![BipedalWalker-v3](https://github.com/sunwuzhou03/rl_learning/BipedalWalker-v3/BipedalWalker-v3.gif)

![BipedalWalkerHardcore-v3](https://github.com/sunwuzhou03/rl_learning/BipedalWalkerHardcore-v3/BipedalWalkerHardcore-v3.gif)

