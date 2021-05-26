import logging, os
import numpy as np
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.keras.backend.clear_session()
import networkx as nx
import cv2
from matplotlib import pyplot as plt
from datetime import datetime

def ppo(agent, states, actions, rewards, states_new, not_dones, log_probs, optimizer, mse, gamma, clip_ratio, beta=1):

    policy_net = [v for v in agent.model.trainable_variables if 'policy' in v.name]
    value_net = [v for v in agent.model.trainable_variables if 'value' in v.name]

    p_ind = [('policy' in v.name) for v in agent.model.trainable_variables]
    v_ind = [('value' in v.name) for v in agent.model.trainable_variables]

    policy_losses = tf.boolean_mask(agent.model.losses, p_ind)
    value_losses = tf.boolean_mask(agent.model.losses, v_ind)
    rewards = np.expand_dims(rewards, axis=-1)
    not_dones = np.expand_dims(not_dones, axis=-1)
    v_new_states = agent.v(states_new).numpy()

    with tf.GradientTape() as tape:
        log_prop_on, entropy = agent.log_prob(states, actions, return_entropy=True)
        advantage = (rewards + (beta * entropy) + (not_dones * v_new_states) - agent.v(states)).numpy()
        ratio = tf.exp(log_prop_on - log_probs)

        mask_greater = tf.greater_equal(advantage, 0)
        mask_lesser = tf.less(advantage, 0)
        ad_greater = tf.minimum(ratio, 1+clip_ratio) * advantage
        ad_lesser = tf.maximum(ratio, 1-clip_ratio) * advantage
        actor_loss = tf.where(mask_greater, ad_greater, advantage)
        actor_loss = -tf.where(mask_lesser ,ad_greater, actor_loss) + tf.reduce_sum(policy_losses)

    gradients_actor = tape.gradient(actor_loss, policy_net)
    gradients_actor = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients_actor]
    optimizer.apply_gradients(zip(gradients_actor, policy_net))
    targets = (rewards + beta * entropy) + not_dones * gamma * v_new_states
    # train valie network
    with tf.GradientTape() as tape:
        v = agent.v(states)
        critic_loss = mse(v, targets) + tf.reduce_sum(value_losses)
    gradients_critic = tape.gradient(critic_loss, value_net)
    gradients_critic = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients_critic]
    optimizer.apply_gradients(zip(gradients_critic, value_net))
    return actor_loss, critic_loss, agent


def ppo_adv(agent, regret, antag_not_done, old_env_state, wrapper, seq_lenght, optimizer, mse, gamma, clip_ratio, beta=1):

    policy_net = [v for v in agent.trainable_variables if 'policy' in v.name]
    value_net = [v for v in agent.trainable_variables if 'value' in v.name]

    p_ind = [('policy' in v.name) for v in agent.trainable_variables]
    v_ind = [('value' in v.name) for v in agent.trainable_variables]

    policy_losses = tf.boolean_mask(agent.losses, p_ind)
    value_losses = tf.boolean_mask(agent.losses, v_ind)
    regret = np.expand_dims(regret, axis=-1)
    antag_not_done  = np.expand_dims(antag_not_done, axis=-1)
    # init lstm states
    # compute actions, compute new states, compute v new states
    rec_states_p = agent.initialize_zero_states(1)
    rec_states_v = agent.initialize_zero_states(1)
    wrapper.reset()

    for t in range(seq_lenght):
        action, new_rec_states_p = agent.get_action(old_env_state, t, rec_states_p)
        v, new_rec_states_v = agent.get_v(old_env_state, t, rec_states_v)
        wrapper.wrap(action, t)
        new_env_state = wrapper.return_state()
        v_new_state, _ = agent.get_v(new_env_state, t, new_rec_states_v)

        with tf.GradientTape() as tape:
            log_prop_on, entropy = agent.log_prob(old_env_state, t, rec_states_p, action, return_entropy=True)
            advantage = (regret + (beta * entropy) + (antag_not_done * v_new_state) - v.numpy())
            ratio = tf.exp(log_prop_on - (1/100))

            mask_greater = tf.greater_equal(advantage, 0)
            mask_lesser = tf.less(advantage, 0)
            ad_greater = tf.minimum(ratio, 1+clip_ratio) * advantage
            ad_lesser = tf.maximum(ratio, 1-clip_ratio) * advantage
            actor_loss = tf.where(mask_greater, ad_greater, advantage)
            actor_loss = -tf.where(mask_lesser ,ad_greater, actor_loss) + tf.reduce_sum(policy_losses)
        gradients_actor = tape.gradient(actor_loss, policy_net)
          # Clip-by-value on all trainable gradients
        #gradients_actor = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients_actor]
        optimizer.apply_gradients(zip(gradients_actor, policy_net))

        targets = (regret + 0.01 * entropy) + antag_not_done * gamma * v_new_state
        # train valie network
        with tf.GradientTape() as tape:
            v, _ = agent.get_v(old_env_state, t, rec_states_v)
            critic_loss = mse(v, targets) + tf.reduce_sum(value_losses)
        gradients_critic = tape.gradient(critic_loss, value_net)
        # Clip-by-value on all trainable gradients
        #gradients_critic = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients_critic]
        optimizer.apply_gradients(zip(gradients_critic, value_net))

        old_env_state = new_env_state
        rec_states_p = new_rec_states_p
        rec_states_v = new_rec_states_v

    return actor_loss, critic_loss, agent


def compute_regret(protagonist_rewards, antagonist_rewards):
    antagonist_rewards = tf.cast(antagonist_rewards,  dtype=tf.float32)
    protagonist_rewards = tf.cast(protagonist_rewards, dtype=tf.float32)

    ant_max = tf.reduce_max(antagonist_rewards,-1)
    prot_mean = tf.reduce_mean(protagonist_rewards, -1)
    regret = tf.abs(ant_max - prot_mean)

    antag_not_done = tf.cast(tf.math.less(ant_max, 0), tf.float32)
    return regret, antag_not_done


def compute_adv_reward(protagonist_rewards, antagonist_rewards, env_kwargs, budget_weight = 0.5):
    """
    Regret = difference betwann max reward of antagonis tand average of protagonist over (all) trajectories
    """
    ant_max = tf.reduce_max(antagonist_rewards,-1)
    prot_mean = tf.reduce_mean(protagonist_rewards, -1)
    regret = tf.abs(ant_max - prot_mean)
    regret = tf.cast(regret, dtype=tf.float32)
    antag_not_done = tf.cast(tf.math.less(ant_max, 0), tf.float32)
    shortest_path = compute_shortest_path(env_kwargs)
    block_budget = compute_adversary_block_budget(shortest_path, budget_weight, ant_max)
    print("block_budget ", block_budget)
    print("regret", regret)
    reward = regret + block_budget
    return reward,  antag_not_done
# budget with gradient flow to add to regret for getting gradients

# what is the block budget weight?

def compute_adversary_block_budget(budget, budget_weight, antag_r_max):
    """Compute block budget reward based on antagonist score."""
    # If block_budget_weight is 0, will return 0.

    weighted_budget = budget * budget_weight
    antag_didnt_score = tf.cast(tf.math.less(antag_r_max, 0), tf.float32)
    print("antagonist scored: ", not(antag_didnt_score))

    # Number of blocks gives a negative penalty if the antagonist didn't score,
    # else becomes a positive reward.
    block_budget_reward = antag_didnt_score * -weighted_budget + \
        (1 - antag_didnt_score) * weighted_budget
    return block_budget_reward

def compute_shortest_path(env_kwargs):
    height = env_kwargs["height"]
    width = env_kwargs["width"]
    start = tuple(env_kwargs["start_position"])
    end = tuple(env_kwargs["reward_position"])
    blocks = env_kwargs["block_position"]

    graph = nx.grid_graph([height, width])
    for i in set(tuple(b) for b in blocks):
        print("block:" , i)
        print("start", start, "end", end)
        if (i==start) or (i==end):
            print("not removing node")
            pass
        else:
            print("removing")
            graph.remove_node(i)

    has_path = nx.has_path(
        graph,
        source=start,
        target=end)

    if has_path:
        # Compute shortest path
        shortest_path_length = nx.shortest_path_length(
          graph,
          source=start,
          target=end)
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      shortest_path_length = (self.width * self.height) + 1
    print("environment has path: ", has_path)

    return shortest_path_length

def save_model(model, epoch, path, model_name="model"):
    time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    full_path = f"{path}/models/{model_name}_{epoch}_{time_stamp}"
    print(f"saving model {model_name}")
    model.save_weights(full_path)

def save_env_state(state, epoch, path):
    plt.clf()
    full_path = f"{path}/env_states/env_state_epoch_{epoch}"
    im = plt.imshow(state)
    plt.savefig(full_path)
    plt.clf()
