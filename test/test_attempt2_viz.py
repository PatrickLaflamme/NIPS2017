import numpy as np
import tensorflow as tf
import opensim as osim
import sys
from osim.env import RunEnv

sys.path.append('../model')

import attempt2

client = RunEnv(visualize=True)
# Create environment
observation = client.reset(difficulty=2)

with tf.Session() as session:

    model = attempt2.ActorCriticDDPG(session,
                                    tf.train.GradientDescentOptimizer(learning_rate=0.01),
                                    41,
                                    18,
                                    tf.train.Saver)

    model.restore("../model_attempt2_train_1/")

    for i in range(1500):

        [observation, reward, done, info] = client.step(model.gen_actions(observation))

        print(reward)

        if done:
            observation = client.reset(difficulty=2)
            if not observation:
                break
