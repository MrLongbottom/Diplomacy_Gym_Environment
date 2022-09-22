from diplomacy_gym_environment import DiplomacyEnvironment
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# This is a template setup on how to actually use the Gym environment I made, by setting up a RL agent.


def random_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = [random.choice(env.game.get_all_possible_orders()[loc]) for loc in
                               env.game.get_orderable_locations(power_name)]
    return actions


def random_nn_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = np.array([random.random() for _ in env.action_list])
    return actions


def random_vs_nonrandom_move(probs):
    actions = {}
    for power_name in env.game.powers.keys():
        if power_name == 'AUSTRIA':
            actions[power_name] = np.array(probs)
        else:
            actions[power_name] = [random.random() for _ in env.action_list]
    return actions


def generator():
    while not env.game.is_game_done:
        yield


def create_model():
    inputs = layers.Input(shape=(num_inputs,))
    common1 = layers.Dense(num_middle, activation="relu")(inputs)
    common2 = layers.Dense(num_middle, activation="relu")(common1)
    common3 = layers.Dense(num_middle, activation="relu")(common2)
    common4 = layers.Dense(num_middle, activation="relu")(common3)
    common5 = layers.Dense(num_actions, activation="sigmoid")(common4)

    return keras.Model(inputs=inputs, outputs=common5)


if __name__ == '__main__':
    # My own custom-made gym environment
    env = DiplomacyEnvironment(prints=False, render_path='maps/')

    # seed for reproducibility
    seed = 42

    # setting Epsilon (exploration vs. exploitation parameter)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = (epsilon_max - epsilon_min)

    batch_size = 32
    max_steps_per_episode = 10000

    # size of NN layers
    num_inputs = env.observation_space.n
    num_actions = env.action_space.shape[0]
    num_middle = 128

    # defining learning model
    model = create_model()
    model_target = create_model()

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    # stats to keep track of
    action_history = []
    action_prob_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    # Number of frames to take random action and observe output
    #epsilon_random_frames = 50000
    epsilon_random_frames = 1000
    # Number of frames for exploration
    #epsilon_greedy_frames = 1000000.0
    epsilon_greedy_frames = 1000
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 100

    while True:
        state = np.array(env.reset())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = random_nn_move()
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                #action = tf.argmax(action_probs[0]).numpy()
                action = random_vs_nonrandom_move(action_probs[0])
                print('making actions')

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment

            state_next, reward, done, info = env.step(action)
            episode_reward += reward[0]

            reward = reward[0]
            done = done[0]
            state_next = np.array(state_next[0])
            action_mask = np.zeros(num_actions)
            action_mask[info['AUSTRIA']] = 1
            action_mask.astype('float32')

            # Save actions and states in replay buffer
            action_history.append(action_mask)
            action_prob_history.append(action['AUSTRIA'])
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                # updated mask to just be probabilities
                masks = tf.convert_to_tensor(action_sample, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1
        print("episode: {}, total reward: {:.2f}".format(episode_count, episode_reward))

        if episode_count >= 1000:
            model.save('model/test_model')
            break
