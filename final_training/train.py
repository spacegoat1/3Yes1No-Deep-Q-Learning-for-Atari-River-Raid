import gym
import numpy as np
from collections import deque
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from custom_wrappers import *
from utils import *
from build_model import build_dq_model
from memory import Memory

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def choose_action(q_net, state, epsilon, num_actions):
    if epsilon > np.random.rand(1)[0]:
        action = np.random.choice(num_actions)
    else:
        state_t = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.argmax(q_net(state_t, training=False)[0]).numpy()
    return action



# Using the wrappers for the environment
env = gym.make("ALE/Riverraid-v5")
env = ObservationWrapper(RewardWrapper(ActionWrapper(ConcatObs(FireResetEnv(env), k=4, DEATH_REWARD=1000)), CLIP=False, SCALE=True), GRAYSCALE=True, NORMALIZE=True)
obs = env.reset()


# Build the model
hidden_size = 640
num_actions = 18
q_net = build_dq_model(input_shape=obs.shape, hidden_size=hidden_size, num_actions=num_actions)
q_net_target = build_dq_model(input_shape=obs.shape, hidden_size=hidden_size, num_actions=num_actions)
loss_func = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
experience = Memory(length=10000)


# Defining hyperparameters
discount_factor = 0.99
batch_size = 32
n_steps_episode = 10000
rolling_reward = 0
episode_ix = 0
observation_ix = 0
random_till = 3000
eps_decay = 0.9995
epsilon = 1.0
eps_threshold = 0.1
weight_update_freq = 10
target_update_freq = 500
n_episodes = 50000


# Kick off training
write_to_log("Starting Training for new implementation", include_blank_line=True)
reward_history = np.zeros(n_episodes)
num_steps_history = np.zeros(n_episodes)
SWITCHED_TO_GREEDY = False



while episode_ix < n_episodes:
    state = np.array(env.reset())
    ep_total_reward = 0

    if episode_ix in [500, 2000, 5000, 10000, 20000, 50000]:
        # print("Episode: ", episode_ix)
        write_to_log("Saving model at " + str(episode_ix))
        curr_model_name = "curr_model_rev_" + str(episode_ix)
        curr_model_name_target = "curr_model_rev_target_" + str(episode_ix)
        save_models_and_arrays(curr_model_name, curr_model_name_target, 
                            q_net, q_net_target,
                            reward_history[:episode_ix], 
                            num_steps_history[:episode_ix])

    for _ in range(1, n_steps_episode):
        observation_ix += 1

        if observation_ix < random_till:
            # Still exploring, not following policy
            action = np.random.choice(num_actions)
        else:
            # Take action as suggested by policy
            if not SWITCHED_TO_GREEDY:
                write_to_log("Switching to Greedy, at observation ", observation_ix)
                SWITCHED_TO_GREEDY = True
            action = choose_action(q_net, state, epsilon, num_actions)

        # Take action and get new observations
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        ep_total_reward += reward

        # Update Experience with experience tuple and then update state
        experience.append((state, state_next, action, reward, done))
        state = state_next

        # Decay epsilon till threshold
        epsilon = max(epsilon*eps_decay, eps_threshold)

        if len(experience) > batch_size and observation_ix % weight_update_freq == 0:

            state_sample, state_next_sample, rewards_sample, action_sample, done_sample = \
                experience.sample_memory(batch_size)

            # Calculate Q values as sum of reward and discounted future rewards and handle final state
            future_rewards = q_net_target.predict(state_next_sample)
            updated_q_vals = rewards_sample + (discount_factor * tf.reduce_max(future_rewards, axis=1))
            updated_q_vals = updated_q_vals * (1 - done_sample) - done_sample

            masked_action = tf.one_hot(tf.cast(action_sample, dtype=tf.int32), num_actions)

            with tf.GradientTape() as tape:
                q_vals = q_net(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_vals, masked_action), axis=1)
                loss = loss_func(updated_q_vals, q_action)

            # Backpropagation
            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        if observation_ix % target_update_freq == 0:
            # Update Target QNet
            q_net_target.set_weights(q_net.get_weights())

        if done:
            break

    reward_history[episode_ix] = ep_total_reward
    rolling_reward = np.mean(reward_history[:episode_ix])

    episode_ix += 1



# Save final
curr_model_name = "final_curr_rev_model_" + str(episode_ix)
curr_model_name_target = "final_curr_model_rev_target_" + str(episode_ix)
save_models_and_arrays(curr_model_name, curr_model_name_target, 
                    q_net, q_net_target,
                    reward_history, num_steps_history)
write_to_log("Training for new implementation complete")
