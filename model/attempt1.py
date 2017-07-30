import numpy as np
import tensorflow as tf

# highly generalized function to perform a feedforward sweep of a fully connected ANN.
def multilayer_perceptron(x, weights, biases, activation_func = tf.nn.relu, dropout_keep_prob=0.8, noise = False):

    layer_output = x

    num_keys = len(weights.keys())

    for i, layer in enumerate(sorted(weights.keys())):

        layer_activation = tf.add(tf.matmul(layer_output, weights[layer]), biases[layer], name="activation" + layer)

        if i < num_keys-1:

            layer_output = activation_func(layer_activation, name = "output" + layer)

            layer_output = tf.nn.dropout(layer_output,dropout_keep_prob)

            if noise:

                mean, var = tf.nn.moments(layer_output, axes=[1])

                layer_output = tf.add(layer_output, tf.random_normal(tf.shape(layer_output), stddev = tf.sqrt(var)/100))

        else:

            layer_output = layer_activation

        tf.summary.histogram("output" + layer, layer_output)

    return layer_output

# a generalized function whose purpose is to initialize the weights for a fully connect ANN.
def gen_mlp_weights(num_input, num_output, n_hidden, n_layers, name=''):

    if len(n_hidden) == 1:
        n_hidden = [num_input] + [n_hidden] * n_layers + [num_output]

    elif len(n_hidden) != n_layers:
        raise ValueError("n_hidden must be scalar or of length n_layers")

    else:
        n_hidden = [num_input] + n_hidden + [num_output]

    weights = dict()
    biases = dict()

    for layer_number in range(n_layers+1):

        weights[str(layer_number)+'_'+name] = tf.Variable(tf.random_normal([n_hidden[layer_number], n_hidden[layer_number+1]],stddev=1/(5*n_hidden[layer_number+1])), name = name + str(layer_number))

        biases[str(layer_number)+'_'+name] = tf.Variable(tf.random_normal([n_hidden[layer_number+1]],stddev=1/(5*n_hidden[layer_number+1])), name = name + str(layer_number) + '_bias')

    return [weights, biases]

# Specialized function to generate weights for the state prection sub-model.
def gen_state_weights(num_actions, num_states, n_hidden, n_layers):

    [weights, biases] = gen_mlp_weights(num_actions + num_states, 1, n_hidden, n_layers, name = "state_prediction")

    return [weights, biases]

# Specialized function to perform a feed-forward sweep of the state prediction sub-model.
def state_prediction(action_list, state_list, weights, biases):

    input_values = tf.concat([action_list, state_list],-1)

    prediction = multilayer_perceptron(input_values, weights, biases)

    return prediction

# Specialized function to generate weights for the action generation sub-model.
def gen_action_weights(num_actions, num_states, n_hidden, n_layers):

    [weights, biases] = gen_mlp_weights(num_states*2, num_actions, n_hidden, n_layers, name = "action_generation")

    return [weights, biases]

# Specialized function to perform a feed-forward sweep of the action generation sub-model
def action_generation(state_list, state_outcome, weights, biases):

    input_values = tf.concat([state_list, tf.subtract(state_outcome, state_list)],-1)

    next_action = multilayer_perceptron(input_values, weights, biases)

    next_action = tf.sigmoid(next_action)

    return next_action

if __name__ == "__main__":

    from osim.env import RunEnv
    import sys
    from joblib import Parallel, delayed
    import multiprocessing

    env = RunEnv(visualize=False)
    env.reset(difficulty=0)

    num_actions = 18
    num_states = 41

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

    [weights['state'], biases['state']] = gen_state_weights(num_actions, num_states, n_hidden_state, n_layers_state)

    [weights['action'], biases['action']] = gen_action_weights(num_actions, num_states, n_hidden_action, n_layers_action)

    actions_list = action_generation(state_list, next_state, weights['action'], biases['action'])

    state_pred = state_prediction(action_list, state_list, reward, weights['state'], biases['state'])

    state_cost = tf.reduce_mean(tf.square(tf.concat([next_state, reward],-1) - state_pred), name = 'cost')
    state_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = state_optimizer.minimize(state_cost)

    action_cost = -(tf.log(actions_list)*reward)
    action_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_state = action_optimizer.minimize(action_cost)

    if sys.argv[1] == 'train_state':

        num_rounds = 5000
        num_steps = 500
        display_step = 100

        action_space_array = np.array([1]*4 + [0]*14)

        choice = np.random.choice
        probs = [.5,.5]

        shuffle = np.random.shuffle

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            i = 0

            for iterval in range(num_rounds):

                env.reset(difficulty=0)

                shuffle(action_space_array)
                old_observation, old_reward, done, info = env.step(action_space_array)

                tot_loss = 0

                for step in range(num_steps):

                    if choice([True, False], p=probs):
                        shuffle(action_space_array)

                    observation, obs_reward, done, info = env.step(action_space_array)

                    pred_obs, loss, _ = sess.run([state_pred, state_cost, update_state],
                                                feed_dict={
                                                    action_list: [action_space_array],
                                                    state_list: [old_observation],
                                                    reward: [[obs_reward]],
                                                    next_state: [observation]
                                                        })

                    tot_loss += loss

                if iterval % display_step == 0:

                    print("iter="+str(iterval) + ", average loss=" + str(tot_loss/display_step))
