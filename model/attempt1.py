import numpy as np
import tensorflow as tf

# highly generalized function to perform a feedforward sweep of a fully connected ANN.
def multilayer_perceptron(x, weights, biases, activation_func = tf.nn.relu):

    layer_input = x

    for layer in weights.keys():

        layer_activation = tf.add(tf.matmul(layer_input, weights[layer]), biases[layer], name="activation" + layer)

        layer_output = activation_func(layer_activation, name = "output" + layer)

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

    for layer_number in range(n_layers):

        weights[layer_number+'_'+name] = tf.Variable(tf.random_normal([n_hidden[layer_number], n_hidden[layer_number+1]]), name = name + str(layer_number))

        biases[layer_number+'_'+name] = tf.Variable(tf.random_normal(n_hidden[layer_number+1]))

    return [weights, biases]

# Specialized function to generate weights for the state prection sub-model.
def gen_state_weights(num_actions, num_states, n_hidden, n_layers):

    [weights, biases] = gen_mlp_weights(num_actions + num_states, num_states, n_hidden, n_layers, name = "state_prediction")

    return [weights, biases]

# Specialized function to perform a feed-forward sweep of the state prediction sub-model.
def state_prediction(action_list, state_list, state_outcome, weights, biases):

    input_values = tf.concat([action_list, state_list],-1)

    prediction = multilayer_perceptron(input_values, weights, biases)

    return prediction

# Specialized function to generate weights for the action generation sub-model.
def gen_action_weights(num_actions, num_states, n_hidden, n_layers):

    [weights, biases] = gen_mlp_weights(num_actions + num_states*2, num_actions, n_hidden, n_layers, name = "action_generation")

    return [weights, heights]

# Specialized function to perform a feed-forward sweep of the action generation sub-model
def action_generation(state_list, state_outcome, weights, biases):

    input_values = tf.concat(state_list, tf.subtract(state_outcome, state_list))

    next_action = multilayer_perceptron(input_values, weights, biases)

if __name__ = "__main__":

    num_actions = 18
    num_states = 41

    n_layers_state = 2
    n_hidden_state = [100,100]

    n_layers_action = 3
    n_hidden_action = [400,400,400]

    weights = dict()
    biases = dict()

    actions = tf.placeholder(tf.float32, [num_actions], name='action_list')
    states = tf.placeholder(tf.float32, [num_states], name = 'state_list')
    pred_states = tf.placeholder(tf.float32, [num_states], name = 'pred_states')
    state_error = tf.subtract(states, pred_states, name='state_error')


    [weights['state'], biases['state']] = gen_state_weights(num_actions, num_states, n_hidden_state, n_layers_state)

    [weights['action'], biases['action']] = gen_action_weights(num_actions, num_states, n_hidden_action, n_layers_action)
