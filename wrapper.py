import numpy as np
import tensorflow as tf
class Wrapper:
    def __init__(self, width, height, start_pos, reward_pos, seq_lenght, env_class):

        self.width = width
        self.height = height
        self.n_states = height*width
        self.start_pos = start_pos
        self.reward_pos = reward_pos
        self.seq_lenght = seq_lenght
        self.env_class = env_class
        self.env_kwargs = {
            "start_position" : [],
            "reward_position" : [],
            "block_position" : [],
            "height" : self.height,
            "width": self.width
        }

    # transforms the output of the adv vector into coodinates for the env

    def wrap(self, net_out, t):

        # onehotify
        net_out = tf.one_hot(tf.argmax(net_out, axis=-1), depth=self.n_states)
        net_out = np.reshape(np.squeeze(net_out), (self.height, self.width))
        net_out = np.squeeze(np.argwhere(net_out==1))
        if t==self.start_pos:
            self.env_kwargs["start_position"] = net_out
        elif t==self.reward_pos:
            while (net_out == self.env_kwargs["start_position"]).all():
                # random reward positioning
                print("choosing random reward position")
                net_out = self.generate_random_position()

            self.env_kwargs["reward_position"] = net_out

        else:
            if np.isin(net_out,self.env_kwargs["block_position"]).all():
                pass
            else: self.env_kwargs["block_position"].append(net_out)
        #print("current env kwargs")
        #print(self.env_kwargs)

        #print("t: ", t, "env kwargs", self.env_kwargs)

    def generate_random_position(self):
        out = np.random.uniform(0,1,self.n_states)
        out = tf.one_hot(tf.argmax(out, -1), depth=self.n_states)
        out = np.reshape(out, (self.height, self.width))
        out = np.squeeze(np.argwhere(out==1))
        return out

    def return_state(self):
        #print("returning state: env kwargs")
        #print(self.env_kwargs)
        env = self.env_class(**self.env_kwargs)
        state = env.reset()
        #print("state")
        #print(state)

        return env.reset()


    def reset(self):
        self.env_kwargs = {
            "start_position" : [],
            "reward_position" : [],
            "block_position" : [],
            "height" : self.height,
            "width": self.width
        }


    def wrap_all(self,network_out):
        """depricated"""
        coordinates = []
        # net out has the shape (seq_lenght, width x height)
        for s in network_out:
            s = tf.one_hot(tf.argmax(s, axis=-1), depth=self.n_states)
            s = np.squeeze(s)
            s = np.reshape(s, (self.height, self.width))
            coordinates.append(np.squeeze(np.argwhere(s==1)))

        start = np.squeeze(coordinates.pop(self.start_pos))
        print('s', start)
        reward = np.squeeze(coordinates.pop(self.reward_pos))

        env_kwargs = {
            "start_position" : start,
            "reward_position" : reward,
            "block_position" : coordinates}

        return env_kwargs
