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

    num_actions = 18
    num_states = 41

    batch_size = 8

    n_layers_state = 2
    n_hidden_state = [512,512]

    n_layers_action = 3
    n_hidden_action = [400,400,400]

    weights = dict()
    biases = dict()

    action_list = tf.placeholder(tf.float32, [None,num_actions], name='action_list')
    state_list = tf.placeholder(tf.float32, [None,num_states], name = 'state_list')
    reward = tf.placeholder(tf.float32, [None,1], name='reward')
    next_state = tf.placeholder(tf.float32, [None,num_states], name = 'next_state')

    [weights['state'], biases['state']] = model.gen_state_weights(num_actions, num_states, n_hidden_state, n_layers_state)

    #[weights['action'], biases['action']] = model.gen_action_weights(num_actions, num_states, n_hidden_action, n_layers_action)

    #actions_list = model.action_generation(state_list, next_state, weights['action'], biases['action'])

    state_pred = model.state_prediction(action_list, state_list, reward, weights['state'], biases['state'])

    state_cost = tf.reduce_mean(tf.square(tf.concat([next_state, reward],-1) - state_pred), name = 'cost')
    state_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = state_optimizer.minimize(state_cost)

    #action_cost = -(tf.log(actions_list)*reward)
    #action_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #update_state = action_optimizer.minimize(action_cost)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", state_cost)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    if sys.argv[1] == 'train_state':

        save_file = "state/sate_vars"

        logs_path = "state_training_log"

        num_rounds = 500
        num_steps = 500
        display_step = 5

        data_save = [[None]*num_steps]*num_rounds

        action_space_array = [np.array([1]*4 + [0]*14)]*batch_size

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

                    pool.map(env_reset, [0]*batch_size)

                    output = pool.map(env_step, action_space_array)

                    action_space_array, old_observation, obs_reward = [val[0] for val in output], [val[1] for val in output], [[val[2]] for val in output]

                    tot_loss = 0

                    num_relived = 0

                    if iterval > 99:

                        num_relived = 20

                        relive = data_save[choice(iterval-1, num_relived)]

                        relived_actions = [val for val in batch[0] for batch in relive]

                        relived_old_obs = [val for val in batch[1] for batch in relive]

                        relived_obs = [val for val in batch[2] for batch in relive]

                        relived_reward = [val for val in batch[3] for batch in relive]

                    for step in range(num_steps):

                        output = pool.map(env_step, action_space_array)

                        action_space_array, observation, obs_reward = [val[0] for val in output], [val[1] for val in output], [[val[2]] for val in output]

                        data_save[iterval][step] = [action_space_array, old_observation, observation, obs_reward]

                        if iterval > 99:
                            all_actions = action_space_array.append(relived_actions[step])

                            all_obs = observation.append(relived_obs[step])

                            all_old_obs = old_observation.append(relived_old_obs[step])

                            all_reward = obs_reward.append(relived_reward[step])

                        else:
                            all_actions = action_space_array

                            all_obs = observation

                            all_old_obs = old_observation

                            all_reward = obs_reward

                        pred_obs, loss, _, summary = sess.run([state_pred, state_cost, update_state, merged_summary_op],
                                                    feed_dict={
                                                        action_list: all_actions,
                                                        state_list: all_old_obs,
                                                        reward: all_reward,
                                                        next_state: all_obs
                                                            })

                        summary_writer.add_summary(summary, iterval * num_steps + step)

                        old_observation = observation

                        tot_loss += loss

                    if iterval % display_step == 0:

                        print("iter="+str(iterval) + ", average loss=" + str(tot_loss/(display_step*(batch_size+num_relived))))

                        saver.save(sess, save_file, global_step=int(iterval/display_step))

        with open('outfile.pkl', 'wb') as fp:
            pickle.dump(data_save, fp)


        print('Training complete! ')
