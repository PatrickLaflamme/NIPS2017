import tensorflow as tf

def multilayer_perceptron(x, weights, biases, activation_func):

    layer_input = x

    for layer in weights.keys():

        layer_activation = tf.add(tf.matmul(layer_input, weights[layer]), biases[layer], name="activation" + layer)

        layer_output = activation_func(layer_activation, name = "output" + layer)

        tf.summary.histogram("output" + layer, layer_output)

    return layer_output

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


def forward_prediction(action_list, state_list, state_outcome):

    input_values = tf.concat([action_list, state_list],-1)
