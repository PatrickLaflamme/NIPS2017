import numpy as np
import tensorflow as tf
import random
from osim.env import RunEnv
from multiprocessing import Pool

env = RunEnv(visualize=False)
env.reset(difficulty=0)

class mlp_network(object):
    # create an MLP network. This is a very simple ANN architecture. However, this function is general enough to construct any simply MLP network.
    def __init__(self, num_input,
                       num_output,
                       n_layers,
                       n_hidden,
                       name,
                       activation_func = tf.nn.relu,
                       out_activation_function = tf.identity):

        self.weights, self.biases = self.create_variables(name, num_input, num_output, n_layers, n_hidden)

        self.out_activation_function = out_activation_function

        self.name = name

        self.activation_func = activation_func

    def create_variables(self, name, num_input, num_output, n_layers, n_hidden):

        assert type(n_hidden) in [type([1]), type((1)),type(np.array(1))], "n_hidden must be a list or tuple"

        with tf.name_scope(name):

            weights = dict()
            biases = dict()

            if len(n_hidden) == 1:
                n_hidden = [num_input] + [n_hidden] * n_layers + [num_output]

            elif len(n_hidden) != n_layers:
                raise ValueError("n_hidden must be an array of length 1 or of length n_layers")

            else:
                n_hidden = [num_input] + n_hidden + [num_output]

            for layer_number in range(n_layers+1):

                weights[name+ "_" + str(layer_number)] = tf.Variable(tf.random_normal([n_hidden[layer_number], n_hidden[layer_number+1]],stddev=1/(5*n_hidden[layer_number+1])), name = name + str(layer_number) + '_weights')

                biases[name+ "_" + str(layer_number)] = tf.Variable(tf.random_normal([n_hidden[layer_number+1]],stddev=1/(5*n_hidden[layer_number+1])), name = name + str(layer_number) + '_bias')

            return [weights, biases]

    def forward_pass(self, x, dropout_keep_prob=0.8, noise=None):

        layer_output = x

        weights = self.weights
        biases = self.biases

        num_keys = len(weights.keys())

        for i, layer in enumerate(sorted(weights.keys())):

            layer_activation = tf.add(tf.matmul(layer_output, weights[layer]), biases[layer], name="activation" + layer)

            if i < num_keys-1:

                layer_output = self.activation_func(layer_activation, name = "output" + layer)

                layer_output = tf.nn.dropout(layer_output,dropout_keep_prob)

                if noise is not None:

                    mean, var = tf.nn.moments(layer_output, axes=[0,1])

                    layer_output = tf.add(layer_output, tf.random_normal(tf.shape(layer_output), stddev = tf.sqrt(var)*(1/noise)))

            else:

                layer_output = self.out_activation_function(layer_activation)

            tf.summary.histogram(layer + "output", layer_output)

        return layer_output

    def copy(self, weights, biases):

        name = self.name

        for layer_number, layer in enumerate(sorted(weights.keys())):

            self.weights[name+ "_" + str(layer_number)] = tf.identity(weights[layer], name = name + str(layer_number) + '_weights')

            self.biases[name+ "_" + str(layer_number)] = tf.identity(biases[layer], name = name + str(layer_number) + '_biases')


class ActorCriticDDPG(object):

    def __init__(self, session,
                       optimizer,
                       state_dim,
                       num_actions,
                       saver,
                       buffer_size = 10000,
                       reg_param = 0.001,
                       discount_reward = 0.99,
                       max_gradient = 5,
                       update_lag_factor = 0.001):


        self.optimizer = optimizer
        self.session = session
        self.saver = saver

        self.critic_network = mlp_network(num_input = state_dim + num_actions,
                                          num_output = 1,
                                          n_layers = 3,
                                          n_hidden = [1024, 1024, 1024],
                                          name = "critic")
        self.actor_network = mlp_network(num_input = state_dim,
                                          num_output = num_actions,
                                          n_layers = 3,
                                          n_hidden = [1024, 1024, 1024],
                                          name = "actor")

        self.slow_critic = mlp_network(num_input = state_dim + num_actions,
                                          num_output = 1,
                                          n_layers = 3,
                                          n_hidden = [1024, 1024, 1024],
                                          name = "slow_critic")

        self.slow_critic.copy(self.critic_network.weights, self.critic_network.biases)

        self.slow_actor = mlp_network(num_input = state_dim,
                                          num_output = num_actions,
                                          n_layers = 3,
                                          n_hidden = [1024, 1024, 1024],
                                          name = "slow_actor")

        self.slow_actor.copy(self.actor_network.weights, self.actor_network.biases)


        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reg_param = reg_param
        self.max_gradient = max_gradient
        self.discount_reward = discount_reward
        self.update_lag_factor = update_lag_factor

        self.tot_reward = 0

        self.buffer = [None]*buffer_size
        self.buffer_location = 0
        self.full_buffer = False

        self.train_iteration = 0

        self.tot_critic_loss = 0
        self.tot_actor_loss = 0

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

    def create_variables(self):

        with tf.name_scope("model_input"):
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name = 'states')
            self.noise = tf.placeholder(tf.float32, [1], name='noise')

        with tf.name_scope('feed_forward_pass'):

            #for forward pass when generating actions.
            with tf.variable_scope('actor'):
                self.action_estimate = self.actor_network.forward_pass(self.states, noise=self.noise)
                self.chosen_actions = self.actor_network.forward_pass(self.states, name="predictions")

        self.actor_network_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        self.critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")

        with tf.name_scope("compute_pg_gradients"):

            self.taken_actions = tf.placeholder(tf.float32, [None, self.num_actions], name = "taken_actions")
            self.reward = tf.placeholder(tf.float32, [None, 1], name = "known_reward")
            self.result_state = tf.placeholder(tf.float32, [None, self.state_dim], name = 'result_state')

            with tf.variable_scope("Reward_True_Estimate", reuse=True):
                self.predicted_actions = self.slow_actor.forward_pass(self.result_state)
                self.slow_values_estimate = self.slow_critic.forward_pass(tf.concat([self.result_state, self.predicted_actions], -1))
                self.assumed_reward = tf.add(self.reward, tf.multiply(self.discount_reward, self.slow_values_estimate))


            with tf.variable_scope("Critic_Reward_Guess", reuse=True):
                self.estimated_values = self.critic_network.forward_pass(tf.concat([self.states, self.taken_actions], -1))

            # compute critic loss
            self.critic_loss = tf.reduce_mean(tf.square(self.assumed_reward - self.estimated_values))
            self.critic_reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.critic_network_variables])
            self.critic_loss = self.critic_loss + self.reg_param * self.critic_reg_loss

            # compute critic gradients
            self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, self.critic_network_variables)

            # compute actor gradients
            self.actor_loss = tf.reduce_mean(-self.critic_network.forward_pass(tf.concat([self.states, self.action_estimate],-1)))
            self.actor_reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.actor_network_variables])
            self.actor_loss = self.actor_loss + self.reg_param * self.actor_reg_loss
            self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, self.actor_network_variables)

            # collect all gradients
            self.gradients = self.actor_gradients + self.critic_gradients

            # clip gradients
            for i, (grad, var) in enumerate(self.gradients):
                # clip gradients by norm
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            # summarize gradients
            for grad, var in self.gradients:
                tf.summary.histogram(var.name, var)

                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)

            # training update
            with tf.name_scope("train_actor_critic"):
            # apply gradients to update actor network
                self.train_op = self.optimizer.apply_gradients(self.gradients)

            # summarize the process
            self.summarize = tf.summary.merge_all()
            self.saver = self.saver()
            self.no_op = tf.no_op()

    def sim_step(self, action_space_array, previous_steps, update_function, pool = None):

        if pool:

            output = pool.starmap(update_function, zip(action_space_array, previous_steps))

        else:

            output = [update_function(action_space_array[0][0], previous_steps)]

        buffer_values = [tuple(val[0]) for val in output]

        num_save = len(buffer_values)

        #print("output:", output)

        index_end = self.buffer_location + num_save

        reset = False

        if index_end > len(self.buffer):

            reset = True

            index_end = len(self.buffer)

            num_save = index_end - self.buffer_location

        for i in range(num_save):

            self.buffer[self.buffer_location + i] = buffer_values[i]

        if reset:

            self.buffer_location = 0
            self.full_buffer = True

        else:

            self.buffer_location = index_end

        self.previous_steps = [valset[1] for valset in output]

    def sample_action(self, update_function, pool = None):

        observations = [val[0] for val in self.previous_steps]

        actions = self.session.run([self.action_estimate], feed_dict={self.states:observations, self.noise:[random.choice(range(2,100))]
        })

        self.sim_step(actions, self.previous_steps, env_step, pool = pool)

    def train_step(self, batch_size):

        if self.full_buffer:

            sample_idx = random.sample(range(len(self.buffer)), batch_size)

        else:

            sample_idx = random.sample(range(self.buffer_location), batch_size)

        sample_states = [self.buffer[i] for i in sample_idx]

        old_obs, reward, action, obs = [sample[0] for sample in sample_states], [[sample[1]] for sample in sample_states], [sample[2] for sample in sample_states], [sample[3] for sample in sample_states]

        critic_loss, actor_loss, _ = self.session.run([self.critic_loss, self.actor_loss, self.train_op], feed_dict = {self.states: old_obs,
                                     self.reward: reward,
                                     self.taken_actions: action,
                                     self.result_state: obs,
                                     self.noise:[1]})

        self.tot_critic_loss += critic_loss
        self.tot_actor_loss += actor_loss


        # update the slow networks
        for i, layer in enumerate(sorted(self.slow_actor.weights.keys())):

            self.slow_actor.weights[layer] = self.actor_network.weights["actor_"+str(i)]*self.update_lag_factor + (1-self.update_lag_factor) * self.slow_actor.weights[layer]

            self.slow_actor.biases[layer] = self.actor_network.biases["actor_"+str(i)]*self.update_lag_factor + (1-self.update_lag_factor) * self.slow_actor.biases[layer]

        for i, layer in enumerate(sorted(self.slow_critic.weights.keys())):

            self.slow_critic.weights[layer] = self.critic_network.weights["critic_"+str(i)]*self.update_lag_factor + (1-self.update_lag_factor) * self.slow_critic.weights[layer]

            self.slow_critic.biases[layer] = self.critic_network.biases["critic_"+str(i)]*self.update_lag_factor + (1-self.update_lag_factor) * self.slow_critic.biases[layer]

        self.train_iteration += 1

    def save(self, step):
        self.saver.save(self.session, "model_attempt2_train_1/model", global_step = step)

    def restore(self, dir):
        self.saver.restore(self.session, tf.train.latest_checkpoint(dir))

    def gen_actions(self, observation):

        actions = self.session.run([self.chosen_actions], feed_dict={self.states:observation})

        return actions



def env_step(action_space_array, previous_steps, difficulty = 0):

    observation, obs_reward, done, info = env.step(action_space_array)

    full_step = previous_steps[0] + [action_space_array, observation]

    if done:

        env.reset(difficulty = difficulty)

        action_space_array = np.zeros_like(action_space_array)

        observation, obs_reward, done, info = env.step(action_space_array)

    next_step = [observation, obs_reward]

    return [full_step, next_step]

if __name__ == '__main__':

    with tf.Session() as session:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        state_dim = 41
        num_actions = 18

        num_steps = 10000000
        display_step = 250

        batch_size = 256

        action_space_array = [[[0]*18]]

        previous_steps = [[[0]*41,0]]

        saver = tf.train.Saver

        model =  ActorCriticDDPG(session,
                                   optimizer,
                                   state_dim,
                                   num_actions,
                                   saver)

        model.sim_step(action_space_array, previous_steps, env_step)

        for step in range(num_steps):

            model.sample_action(env_step)

            if step < batch_size:

                size  = step + 1

            else:

                size = batch_size

            model.train_step(size)

            if step % display_step == 0:

                print("iter = " + str(step) + ", actor_loss = " + str(model.tot_actor_loss/model.train_iteration) + ", critic_loss = " +
                str(model.tot_critic_loss/model.train_iteration))

                model.save(step)

                model.tot_actor_loss = 0
                model.tot_critic_loss = 0
                model.train_iteration = 0
