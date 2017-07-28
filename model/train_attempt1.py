from osim.env import RunEnv
import sys
from multiprocessing import Pool
import numpy as np

shuffle = np.random.shuffle
probs = [0.4,0.6]
choice = np.random.choice


env = RunEnv(visualize=False)
env.reset(difficulty=0)

def env_set(action_space_array):

    if choice([True, False], p=probs):
        shuffle(action_space_array)

    observation, obs_reward, done, info = env.step(action_space_array)

    return [action_space_array, observation, obs_reward]



if __name__ == "__main__":

    import tensorflow as tf
    import attempt1 as model

    num_actions = 18
    num_states = 41

    batch_size = 8

    n_layers_state = 2
    n_hidden_state = [100,100]

    n_layers_action = 3
    n_hidden_action = [400,400,400]

    weights = dict()
    biases = dict()

    action_list = tf.placeholder(tf.float32, [None,num_actions], name='action_list')
    state_list = tf.placeholder(tf.float32, [None,num_states], name = 'state_list')
    reward = tf.placeholder(tf.float32, [None,1], name='reward')
    next_state = tf.placeholder(tf.float32, [None,num_states], name = 'next_state')

    [weights['state'], biases['state']] = model.gen_state_weights(num_actions, num_states, n_hidden_state, n_layers_state)

    [weights['action'], biases['action']] = model.gen_action_weights(num_actions, num_states, n_hidden_action, n_layers_action)

    actions_list = model.action_generation(state_list, next_state, weights['action'], biases['action'])

    state_pred = model.state_prediction(action_list, state_list, reward, weights['state'], biases['state'])

    state_cost = tf.reduce_mean(tf.square(tf.concat([next_state, reward],-1) - state_pred), name = 'cost')
    state_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = state_optimizer.minimize(state_cost)

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

    if sys.argv[1] == 'train_state':

        save_file = "state/sate_vars"

        log_folder = "state_training_log"

        num_rounds = 5000
        num_steps = 500
        display_step = 100

        action_space_array = [np.array(batch_size, [1]*4 + [0]*14)]*batch_size

        choice = np.random.choice
        probs = [.5,.5]

        shuffle = np.random.shuffle

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

            with Pool(4) as pool:

                for iterval in range(num_rounds):

                    pool.map(env.reset, [0]*batch_size)

                    output = pool.map(env_step, action_space_array)

                    action_space_array, observation, obs_reward = [val[0] for val in output], [val[1] for val in output], [val[2] for val in output]

                    tot_loss = 0

                    for step in range(num_steps):

                        output = pool.map(env_step, action_space_array)

                        action_space_array, observation, obs_reward = [val[0] for val in output], [val[1] for val in output], [val[2] for val in output]

                        pred_obs, loss, _, summary = sess.run([state_pred, state_cost, update_state, merged_summary_op],
                                                    feed_dict={
                                                        action_list: [action_space_array],
                                                        state_list: [old_observation],
                                                        reward: [[obs_reward]],
                                                        next_state: [observation]
                                                            })

                        summary_writer.add_summary(summary, iterval * num_steps + step)

                        tot_loss += loss

                    if iterval % display_step == 0:

                        print("iter="+str(iterval) + ", average loss=" + str(tot_loss/display_step))

                        saver.save(sess, save_file, global_step=iterval/display_step)
