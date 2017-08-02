import numpy as np
import tensorflow as tf
import opensim as osim
import sys
from osim.http.client import Client
from osim.env import RunEnv

sys.path.append('../model')

import attempt2

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = sys.argv[1]

client = Client(remote_base)
# Create environment
observation = client.env_create(crowdai_token)

with tf.Session() as session:

    model = attempt2.ActorCriticDDPG(session,
                                    tf.train.GradientDescentOptimizer,
                                    tf.train.saver,
                                    18,
                                    41)

    model.restore("../model_attempt2_train_1/")

    for i in range(1500):

        [observation, reward, done, info] = client.env_step(model.gen_actions(observation), True)

        print(reward)

        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()
