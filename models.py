

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.keras.backend.clear_session()

import numpy as np
from gridworlds import GridWorld_Global_Multi
from wrapper import Wrapper
import cv2
import networkx as nx
from really.utils import tf_unique_2d


class Gonist(tf.keras.Model):
    def __init__(self, output_units):

        super(Gonist, self).__init__()
        # policy network
        self.h1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, name='policy', padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.h2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, name='policy', padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.h3 = tf.keras.layers.Dense(16, name='policy', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.pooling_p = tf.keras.layers.GlobalAveragePooling2D(name='policy')#self.layer_mu = tf.keras.layers.Dense(output_units, activation=tf.nn.tanh, name='policy')
        #self.layer_sigma = tf.keras.layers.Dense(output_units, activation=tf.nn.tanh, name='policy'))
        self.layer_policy = tf.keras.layers.Dense(output_units, activation=tf.nn.softmax, use_bias=False, name='policy', kernel_regularizer=tf.keras.regularizers.l2())
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(name='policy', gamma_regularizer=tf.keras.regularizers.l2(),  beta_regularizer=tf.keras.regularizers.l2())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(name='policy',  gamma_regularizer=tf.keras.regularizers.l2(),  beta_regularizer=tf.keras.regularizers.l2())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(name='policy',  gamma_regularizer=tf.keras.regularizers.l2(),  beta_regularizer=tf.keras.regularizers.l2())
        # value network

        # policy network
        self.h4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh, name='value', padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.h5 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh, name='value', padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.h6 = tf.keras.layers.Dense(16, activation=tf.nn.tanh, name='value', kernel_regularizer=tf.keras.regularizers.l2(),bias_regularizer=tf.keras.regularizers.l2())
        self.pooling_v = tf.keras.layers.GlobalAveragePooling2D(name='value')
        self.layer_v = tf.keras.layers.Dense(1, use_bias=False, name='value', kernel_regularizer=tf.keras.regularizers.l2())


    def call(self, x):

        output = {}

        # policy network
        x_pol = self.h1(x)
        x_pol = self.batch_norm_1(x_pol)
        x_pol = tf.nn.tanh(x_pol)
        x_pol = self.h2(x_pol)
        x_pol = self.batch_norm_2(x_pol)
        x_pol = tf.nn.tanh(x_pol)
        x_pol = self.pooling_p(x_pol)
        x_pol = self.h3(x_pol)
        x_pol = self.batch_norm_3(x_pol)
        x_pol = tf.nn.tanh(x_pol)
        p = self.layer_policy(x_pol)
        output['policy'] = p

        # value network
        x_v = self.h4(x)
        x_v = self.h5(x_v)
        x_v = self.pooling_v(x_v)
        x_v = self.h6(x_v)
        v = self.layer_v(x_v)
        output["value_estimate"] = v

        return output



class Adversarial(tf.keras.Model):
    def __init__(self, n_states, width, height):

        super(Adversarial, self).__init__()

        self.p1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh, padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='policy', input_shape=[None, height, width, 3])
        self.p2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh,  padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='policy')
        self.p3 = tf.keras.layers.Dense(16, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='policy')
        self.lstm_p = tf.keras.layers.LSTMCell(n_states, activation='softmax', name='policy', kernel_regularizer=tf.keras.regularizers.l1(), recurrent_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())

        self.v1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh, padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='value', input_shape=[None, height, width, 3])
        self.v2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.tanh, padding='same', kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='value')
        self.v3 = tf.keras.layers.Dense(16, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2(), name='value')
        self.lstm_v = tf.keras.layers.LSTMCell(n_states, activation=tf.nn.tanh, name='value', kernel_regularizer=tf.keras.regularizers.l1(), recurrent_regularizer=tf.keras.regularizers.l2(), bias_regularizer=tf.keras.regularizers.l2())
        self.readout_v = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(), name='value')

        self.pooling = tf.keras.layers.GlobalAvgPool2D()
        self.concat = tf.keras.layers.Concatenate()
        self.n_states = n_states
        self.width = width
        self.height = height


        # is suposed to return a sequence of one_hots of size state space
        # sequence of size num_actions
        # readout is lstm with softmax activation
    def initialize_zero_states(self, batch_size):
        return (tf.zeros((batch_size, self.n_states)), tf.zeros((batch_size, self.n_states)))

    def call(self, state, time_step, rec_states_p, rec_states_v):
        output={}

        action, rec_states_p = self.get_action(state, time_step, rec_states_p)
        output['action'] = action
        output['rec_states_p'] = rec_states_p

        value, rec_states_v = self.get_v(state, time_step, rec_states_v)
        output['value'] = value
        output['rec_states_v'] = rec_states_v

        return output

    def get_action(self, state, time_step, rec_states_p):
        state = tf.expand_dims(state, 0)
        time_step = tf.cast(tf.reshape(time_step,(1,-1)), dtype=tf.float32)

        #print("in call state", state)
        #print("in call rec state")
        #print(rec_states_p)
        #print("time step", time_step)

        # policy
        x = self.p1(state)
        #print("first layer")
        #print(x)
        x = self.p2(x)
        #print("second")
        #print(x)
        x = self.pooling(x)
        #print("pooling")
        #print(x)
        x = self.concat([x,time_step])
        #print("concate")
        #print(x)
        #print("last")
        x = self.p3(x)

        action, rec_states_p = self.lstm_p(x, rec_states_p)

        #action += tf.random.normal((1,100)
        #print("action plus noise")
        #print("action")
        #print(action)
        #print("in call new rec")
        #print(rec_states_p)

        if tf.math.reduce_any(tf.math.is_nan(action)): raise ValueError

        return action, rec_states_p

    def get_v(self, state, time_step, rec_states_v):
        state = tf.expand_dims(state, 0)
        time_step = tf.cast(tf.reshape(time_step,(1,-1)), dtype=tf.float32)

        # value
        x = self.v1(state)
        x = self.v2(x)
        x = self.pooling(x)
        x = self.concat([x,time_step])
        x = self.v3(x)

        value, rec_states_v = self.lstm_v(x, rec_states_v)
        value = self.readout_v(value)

        return value, rec_states_v



    def log_prob(self, state, time_step, rec_states_p, action, return_entropy=True):
        """
        only supportet for batch size 1
        action is a one_hot -> have to extract index
        """
        action_index = int(tf.argmax(action, axis=-1).numpy())
        #print("action to get prob")
        #print(action)

        probs, _ = self.get_action(state, time_step, rec_states_p)
        log_prob = tf.math.log(probs[0][action_index])
        log_prob = tf.expand_dims(log_prob, -1)
        #print("action index", action_index)
        #print("original probs")
        #print(probs)
        #print("logs")
        #print(log_prob)
        if return_entropy:
            entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
            entropy = tf.expand_dims(entropy, -1)
            return log_prob, entropy
        else: return log_prob



    def get_shprobs_path_length(self, obstacles):
        coordinates = []
        # net out has the shape (seq_length, width x height)
        for s in obstacles:
            s = np.squeeze(s)
            s = np.reshape(s, (self.height, self.width))
            coordinates.append(tuple(np.squeeze(np.argwhere(s==1)).tolist()))

        start = coordinates.pop(self.start_pos)
        #print('s', start)
        reward = coordinates.pop(self.reward_pos)
        #print('r', reward)

        graph = nx.grid_graph([self.height, self.width])

        for b in set(coordinates):
            if np.all(b==start):
                print("found start passing")
                pass
            elif np.all(b==reward):
                print("found reward passing")
                pass
            else:
                print(f"removing {b}")
                try:
                    graph.remove_node(b)
                except KeyError:
                    print("passing")
                    pass
        has_path = nx.has_path(graph, start, reward)
        print("graph has path:", has_path)
        # large bonus when there is a path -> num_blocks

        # Compute shortest path
        print("uniqe" , tf_unique_2d(tf.where(obstacles)))
        sum = tf.cast(tf.math.greater_equal(tf_unique_2d(tf.where(obstacles)),0), dtype=tf.float32)
        num_obstacles = tf.divide(tf.reduce_sum(sum),2.0) - 2.0
        print("counted obstaclse:", num_obstacles)

          # Impassable environments have a shortest path length 1 longer than
          # abs shortest path
        shortest_path = tf.cast(tf.math.abs(start[0] - reward[0]) + tf.math.abs(reward[1] - start[1]), tf.float32)

        print("shortest path",shortest_path)
        bonus = shortest_path + (tf.cast(has_path, tf.float32) * num_obstacles)

        return tf.expand_dims(bonus,0)

if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    wrapper = Wrapper(10,10,0,1,50)
    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (0, 0),
        "reward_position": (9, 9),
        "block_position": [(0,2),(2,3),(3,0),(4,5)]
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld_Global_Multi(**env_kwargs)
    adv = Adversarial(env.n_states, 50, 10, 10)
    time_step = 0.0

    state = env.reset()
    cv2.imshow("env", state)
    cv2.waitKey(1000)

    obstacles = adv(state, time_step)
    shortest_path_length = adv.get_shortest_path_length(obstacles)
    print(shortest_path_length)
    print(f"shortes pathd {shortest_path_length}")
    env_kwargs = wrapper.wrap(obstacles)
    #print(env_kwargs)
    env = GridWorld_Global_Multi(**env_kwargs)
    state = env.reset()
    cv2.imshow("env",state)
    cv2.waitKey(1000)
