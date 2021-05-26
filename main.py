import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets
from models import Gonist, Adversarial
from gridworld_global_multi0 import GridWorld_Global_Multi
from train import ppo, ppo_adv, compute_regret, compute_adv_reward, compute_shortest_path, save_model, save_env_state
from wrapper import Wrapper


if __name__ == "__main__":
    # default env kwargs

    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 5,
        "width": 5,
        "action_dict": action_dict,
        "start_position": (0, 0),
        "reward_position": (4, 4),
        "block_position": []
    }
    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld_Global_Multi(**env_kwargs)
    model_kwargs = {"output_units": env.action_space.n}

    kwargs = {
        "model": Gonist,
        "environment": GridWorld_Global_Multi,
        "num_parallel": 8,
        "total_steps": 1000,
        "model_kwargs": model_kwargs,
        "env_kwargs" : env_kwargs,
        "returns" : ['log_prob'],
        "action_sampling_type" : "discrete_policy"
    }


    # initilize
    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress/paired_smaller"

    # training hyperparameters
    test_steps = 50
    test_episodes = 10
    epochs = 2000
    sample_size = 1000
    optim_batch_size = 128
    saving_after = 20
    clip_ratio = 0.2
    gamma = 0.85
    inner_epochs = 1
    original_beta=1
    beta=original_beta
    beta_decay=0.9
    old_e=0
    prot_optimizer = tf.keras.optimizers.Adam()
    ant_optimizer = tf.keras.optimizers.Adam()
    adv_optimizer = tf.keras.optimizers.Adam()
    mse = tf.keras.losses.MSE

    protagonist = Gonist(**model_kwargs)
    anatgonist = Gonist(**model_kwargs)
    protagonist(np.expand_dims(env.reset(),0))
    anatgonist(np.expand_dims(env.reset(),0))
    protagonist_weights = protagonist.get_weights()
    antagonist_weights = anatgonist.get_weights()

    # adversarial hyperparameters
    update_env_after = 1
    seq_length= 15
    wrapper = Wrapper(env.width, env.height, 0,1, seq_length, GridWorld_Global_Multi)
    adversarial = Adversarial(env.n_states, width=env_kwargs["width"], height=env_kwargs["height"])
    # initialize weights and env kwargs
    env_state = env.reset()
    rec_states_p = adversarial.initialize_zero_states(1)
    # initialize with call
    adversarial(env_state, -1, rec_states_p, rec_states_p)
    for i in range(seq_length):
        obstacle, rec_states_p = adversarial.get_action(env_state, i, rec_states_p)
        wrapper.wrap(obstacle, i)
        env_state = wrapper.return_state()
    env_kwargs = wrapper.env_kwargs
    manager.set_env(env_kwargs)
    compute_shortest_path(env_kwargs)
    save_env_state(env_state, -1, saving_path)



    optim_keys = ["state", "action", "reward", "state_new", "not_done", "log_prob"]

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["prot_time_steps", "prot_reward", "ant_time_steps","ant_reward", "regret", "adv_critic_loss", "adv_actor_loss"]
    )

    print("test antagonist before training: ")
    manager.set_agent(antagonist_weights)
    manager.test(test_steps, test_episodes=10, do_print=True, evaluation_measure='time_and_reward')


    for e in range(epochs):
        print("optimize portagonist")
        p_actor_losses=[]
        p_critic_losses=[]
        beta = original_beta
        manager.set_agent(protagonist_weights)
        protagonist = manager.get_agent()
        for _ in range(inner_epochs):
            #print("collecting protagonist experience..")

            prot_sample_dict = manager.sample(sample_size, from_buffer=False)
            # create and batch tf datasets
            prot_data_dict = dict_to_dict_of_datasets(prot_sample_dict, batch_size=optim_batch_size)
            #print("optimizing protagonist...")

            for _ in range(5):
                for states, actions, rewards, states_new, not_dones, log_probs in zip(*[prot_data_dict[k] for k in optim_keys]):
                    p_actor_loss, p_critic_loss, protagonist = ppo(protagonist, states, actions, rewards, states_new, not_dones, log_probs, prot_optimizer, mse, gamma, clip_ratio, beta)
                    p_actor_losses.append(np.mean(p_actor_loss, axis=0))
                    p_critic_losses.append(np.mean(p_critic_loss, axis=0))
            beta=beta*beta_decay
            #prot_time_steps, prot_reward = manager.test(test_steps, test_episodes, evaluation_measure='time_and_reward', do_print=True)
            protagonist_weights = protagonist.model.get_weights()
            manager.set_agent(protagonist_weights)

        prot_time_steps, prot_reward = manager.test(test_steps, test_episodes, evaluation_measure='time_and_reward')

        print("optimize antagonist")
        a_actor_losses=[]
        a_critic_losses=[]
        beta = original_beta
        manager.set_agent(antagonist_weights)
        antagonist = manager.get_agent()
        for _ in range(inner_epochs):
            #print("collecting antagonist experience..")

            ant_sample_dict = manager.sample(sample_size, from_buffer=False)
            # create and batch tf datasets
            ant_data_dict = dict_to_dict_of_datasets(ant_sample_dict, batch_size=optim_batch_size)
            #print("optimizing antagonist...")
            a_actor_losses=[]
            a_critic_losses=[]

            for _ in range(5):
                for states, actions, rewards, states_new, not_dones, log_probs in zip(*[ant_data_dict[k] for k in optim_keys]):
                    a_actor_loss, a_critic_loss, antagonist = ppo(antagonist, states, actions, rewards, states_new, not_dones, log_probs, ant_optimizer, mse, gamma, clip_ratio, beta)
                    a_actor_losses.append(np.mean(a_actor_loss, axis=0))
                    a_critic_losses.append(np.mean(a_critic_loss, axis=0))
            beta = beta * beta_decay
            #ant_time_steps, ant_reward = manager.test(test_steps, test_episodes,evaluation_measure='time_and_reward', do_print=True)
            antagonist_weights = antagonist.model.get_weights()
            manager.set_agent(antagonist_weights)
        ant_time_steps, ant_reward = manager.test(test_steps, test_episodes,evaluation_measure='time_and_reward')

        print("optimizing adversarial...")
        #if e%update_env_after == 0:

        #regret, antag_not_done = compute_adv_reward(prot_sample_dict["reward"], ant_sample_dict["reward"], wrapper.env_kwargs)
        regret, antag_not_done = compute_adv_reward(prot_reward, ant_reward, wrapper.env_kwargs)
        wrapper.reset()

        adv_actor_loss, adv_critic_loss, adversarial = ppo_adv(adversarial, regret, antag_not_done, env_state, wrapper, seq_length, adv_optimizer, mse, gamma, clip_ratio, beta=1)
        adv_state = wrapper.return_state()
        env_kwargs = wrapper.env_kwargs
        manager.set_env(env_kwargs)
        save_env_state(adv_state, e, saving_path)
        print("new env kwargs", env_kwargs)

        #print(np.asarray(critic_losses).shape)

        manager.update_aggregator(prot_reward = prot_reward, prot_time_steps=prot_time_steps, ant_time_steps=ant_time_steps, ant_reward = ant_reward, regret=regret, adv_critic_loss=adv_critic_loss, adv_actor_loss=adv_actor_loss)

        # save image of current env


        print(f'epoch ::: {e}')
        #if (e%update_env_after)==0:
        print(f'updating env, regret ::: {regret}   critic loss ::: {np.mean(adv_critic_loss)}  actor loss ::: {np.mean(adv_actor_loss)}')

        print(f'protagonist:  critic_loss ::: {np.mean(p_critic_losses)}    actor loss ::: {np.mean(p_actor_losses)}')
        print(f'average env steps ::: {np.mean(prot_time_steps)}    avg reward ::: {np.mean(prot_reward)}')
        print(f'antagonist:  critic_loss ::: {np.mean(a_critic_losses)}    actor loss ::: {np.mean(a_actor_losses)}')
        print(f'average env steps ::: {np.mean(ant_time_steps)}     avg reward ::: {np.mean(ant_reward)}')

        if e % saving_after == 0:
            # you can save models
            save_model(protagonist.model, e, saving_path, "protagonist")
            save_model(antagonist.model, e, saving_path, "antagonis")
            save_model(adversarial, e, saving_path, "adversarial")

    # and load mmodels
    #manager.load_model(saving_path)
    print("done")
