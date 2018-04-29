import __future__
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
class MyAgent(object):

    def __init__(self):
        print("Initial")

        self.actions = [self.stand,
                        self.move_left,
                        self.move_right,
                        self.jump,
                        self.jump_left,
                        self.jump_right]

    def computeObservation(self, observation):
        # input image dimensions
        img_rows, img_cols = 320, 224

        self.train = observation.reshape(observation.shape[0], 1, img_rows, img_cols)



    def stand(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def jump(self):
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def move_left(self):
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    def move_right(self):
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def jump_right(self):
        return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def jump_left(self):
        return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    def random_action(self):
        action = choice(self.actions)
        return action()




def random_agent():
    env = retro.make(game='SonicTheHedgehog2-Genesis', state='AquaticRuinZone.Act1')
    agent = MyAgent()
    #verbosity = args.verbose - args.quiet
    try:
        while True:
            observation_n = env.reset()
            t = 0
            totrew = 0
            while True:
                ac = agent.random_action()
                observation_n, reward_n, done_n, info = env.step(ac)
                t += 1
                env.render()
    except KeyboardInterrupt:
        exit(0)



if __name__ == '__main__':
    random_agent()
    #SensibleRandomAgent()
    #main()
