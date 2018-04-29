import gym
import retro
import random
import argparse
from functools import partial

from numpy.random import choice
from scipy.stats import poisson

# -----------------------
# Imports used by handwritten digits recognizer
import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# -----------------------

#for game in retro.list_games():
    #print(game, retro.list_states(game))

#reinforcement learning step
class main():
    """
    Script out a set of typical player behaviors and select them with some
    amount of randomness.
    """
    SHORT_JUMP = 1
    MEDIUM_JUMP = 5
    LONG_JUMP = 10

    def __init__(self, poisson_k=20, enable_left=False):
        # init the invironment
        self.env = retro.make(game='SonicTheHedgehog2-Genesis', state='AquaticRuinZone.Act1')



        self.rng = poisson(poisson_k)
        self.time_to_change = 0
        self.jump_timer = 0
        self.jumped = False
        self.current_action = None
        self.actions = [
            self.stand,
            partial(self.jump, self.SHORT_JUMP),
            partial(self.jump, self.MEDIUM_JUMP),
            partial(self.jump, self.LONG_JUMP),
            self.move_right,
            partial(self.jump, self.SHORT_JUMP, move='right'),
            partial(self.jump, self.MEDIUM_JUMP, move='right'),
            partial(self.jump, self.LONG_JUMP, move='right')
        ]
        if enable_left:
            self.actions.extend([
                self.move_left,
                partial(self.jump, self.SHORT_JUMP, move='left'),
                partial(self.jump, self.MEDIUM_JUMP, move='left'),
                partial(self.jump, self.LONG_JUMP, move='left')
            ])

    def stand(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def move_left(self):
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    def move_right(self):
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def jump(self, duration, move=None):
        action = [0] * 12
        if not self.jumped:
            if self.jump_timer == 0:
                self.jump_timer = duration
            else:
                self.jump_timer -= 1
                if self.jump_timer == 0:
                    self.jumped = True
            action[0] = 1
        if move == 'left':
            action[6] = 1
        elif move == 'right':
            action[7] = 1
        return action

    def determine_turn(self, turn, observation_n, j, total_sum, prev_total_sum, reward_n):
        # for every 15 iteration, sum the total observations, and take the average
        # if lower than 0 change the direction

        if j >= 15:
            if total_sum / j == 0:
                turn = True
            else:
                turn = False

            # reset vars
            prev_total_sum = total_sum
            total_sum = 0
            j = 0

        else:
            turn = False

        if observation_n.all() != None:
            # increment counter and reward sum
            j += 1
            total_sum += reward_n
        return turn, j, total_sum, prev_total_sum

    def main(self):

        # init variables
        # num of game operations
        n = 0
        j = 0
        # sum of observations
        total_sum = 0
        reward_n = [0]
        prev_total_sum = 0
        turn = False

    #main logic
        while True:
            observation_n = self.env.reset()
            # increment a counter for number of iterations
            n+=1

            # if at least one iteration is made, check if turn is needed
            if n > 1:
                # if at least one iteration, check if a turn is needed
                if observation_n[0].all is not None:  # 0 for first running Game / Might be multiple
                    # store the reward in the previous score
                    prev_score = reward_n[0]

                    #should we turn?
                    if turn:
                        # pick a random event
                        # where to turn
                        event = random.choice(self.actions)
                        # perform an action
                        action_n = [event for ob in observation_n]
                        # set turn to false
                        turn = False
            elif(~turn):
                # if no turn is needed, go straight
                action_n = [right for ob in observation_n[]]
                print(action_n)

            # if there is an observation, check if turn needed
            if observation_n[0].all() != None:
                turn, j, total_sum, prev_total_sum = self.determine_turn(turn, observation_n[0], j, total_sum, prev_total_sum, reward_n[0])

            #save new variables for each iteration
            observation_n, reward_n, done_n, info = self.env.step(action_n)
            self.env.render()

def random_agent():
    env = retro.make(game='SonicTheHedgehog2-Genesis', state='AquaticRuinZone.Act1')
    agent = SensibleRandomAgent()
    #verbosity = args.verbose - args.quiet
    try:
        while True:
            ob = env.reset()
            t = 0
            totrew = 0
            while True:
                ac = agent.action()
                ob, rew, done, info = env.step(ac)
                t += 1
                env.render()
    except KeyboardInterrupt:
        exit(0)


class SensibleRandomAgent(object):
    """
    Script out a set of typical player behaviors and select them with some
    amount of randomness.
    """
    SHORT_JUMP = 1
    MEDIUM_JUMP = 5
    LONG_JUMP = 10

    def __init__(self, poisson_k=20, enable_left=False):
        self.rng = poisson(poisson_k)
        self.time_to_change = 0
        self.jump_timer = 0
        self.jumped = False
        self.current_action = None
        self.actions = [
            self.stand,
            partial(self.jump, self.SHORT_JUMP),
            partial(self.jump, self.MEDIUM_JUMP),
            partial(self.jump, self.LONG_JUMP),
            self.move_right,
            partial(self.jump, self.SHORT_JUMP, move='right'),
            partial(self.jump, self.MEDIUM_JUMP, move='right'),
            partial(self.jump, self.LONG_JUMP, move='right')
        ]
        if enable_left:
            self.actions.extend([
                self.move_left,
                partial(self.jump, self.SHORT_JUMP, move='left'),
                partial(self.jump, self.MEDIUM_JUMP, move='left'),
                partial(self.jump, self.LONG_JUMP, move='left')
            ])

    def action(self):
        if self.time_to_change <= 0:
            self.current_action = choice(self.actions)
            self.time_to_change = self.rng.rvs()
            self.jumped = False
        self.time_to_change -= 1
        return self.current_action()

    def stand(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def move_left(self):
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    def move_right(self):
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def jump(self, duration, move=None):
        action = [0] * 12
        if not self.jumped:
            if self.jump_timer == 0:
                self.jump_timer = duration
            else:
                self.jump_timer -= 1
                if self.jump_timer == 0:
                    self.jumped = True
            action[0] = 1
        if move == 'left':
            action[6] = 1
        elif move == 'right':
            action[7] = 1
        return action


if __name__ == '__main__':
    #random_agent()
    SensibleRandomAgent()
    main()
