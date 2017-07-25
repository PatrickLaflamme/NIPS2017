from osim.env import RunEnv
import numpy as np

env = RunEnv(visualize)

num_rounds = 50000
num_steps = 500

action_space_array = np.array([1]*4 + [0]*14)

choice = np.random.choice
probs = [.5,.5]

shuffle = np.random.shuffle

for iter in range(num_rounds):

    for step in range(num_steps):

        if choice([True, False], p=probs):
            shuffle(action_space_array)

        observation, reward, done, info = env.step(action_space_array)
