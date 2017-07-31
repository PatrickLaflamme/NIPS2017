from osim.env import RunEnv
import sys
from multiprocessing import Pool
import numpy as np

shuffle = np.random.shuffle
probs = [0.4,0.6]
choice = np.random.choice


env = RunEnv(visualize=False)

env.reset(difficulty=0)

def env_reset(difficulty):

    env.reset(difficulty=difficulty)


def env_step(action_space_array):

    if choice([True, False], p=probs):
        shuffle(action_space_array)

    observation, obs_reward, done, info = env.step(action_space_array)

    return [action_space_array, observation, obs_reward]



if __name__ == "__main__":

    import tensorflow as tf
    import attempt1 as model
    import pickle


    # define the action and state space sizes
    num_actions = 18
    num_states = 41

    # discounting parameter for the Q function rewards
    gamma = 0.99

    # lag parameter for Q' and u'
    tao = 0.001

    # define the batch size
    batch_size = 100

    # Q network size
    n_layers_state = 3
    n_hidden_state = [1024,1024,1024]

    # u network size.
    n_layers_action = 3
    n_hidden_action = [400,400,400]

    weights = dict()
    biases = dict()

    action_list = tf.placeholder(tf.float32, [None,num_actions], name='action_list')
    state_list = tf.placeholder(tf.float32, [None,num_states], name = 'state_list')
    reward = tf.placeholder(tf.float32, [None,1], name='reward')
    next_state = tf.placeholder(tf.float32, [None,num_states], name = 'next_state')

    [weights['state'], biases['state']] = model.gen_state_weights(num_actions, num_states, n_hidden_state, n_layers_state)

    state_pred = model.state_prediction(action_list, state_list, weights['state'], biases['state'])

    
    state_cost = tf.reduce_mean(tf.square(state_pred - reward), name = 'cost')
    state_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = state_optimizer.minimize(state_cost)


    [weights['action'], biases['action']] = model.gen_action_weights(num_actions, num_states, n_hidden_action, n_layers_action)

    actions_list = model.action_generation(state_list, weights['action'], biases['action'])

    action_cost = -(tf.log(actions_list)*reward)
    action_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = action_optimizer.minimize(action_cost)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", state_cost)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    if sys.argv[1] == 'train':


        print('Training complete! ')
